import json
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torchtitan.components.checkpoint import CheckpointManager as TorchTitanCheckpointManager
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


class CheckpointManager(TorchTitanCheckpointManager):
    """Checkpoint manager that avoids unsafe full saves at step 1 and
    tolerates missing keys when resuming from a checkpoint whose model
    architecture has fewer parameters than the current model (e.g. new
    ``q_conv1d`` / ``shortconv`` layers added after the checkpoint was saved).
    """

    def __init__(self, *args: Any, job_config: Any, **kwargs: Any) -> None:
        super().__init__(*args, job_config=job_config, **kwargs)
        self.job_config = job_config
        self._sync_dcp_process_group = None

        checkpoint_config = job_config.checkpoint
        save_backend = getattr(checkpoint_config, "save_backend", "dcp")
        async_mode = getattr(checkpoint_config, "async_mode", "disabled").lower()
        if (
            self.enable_checkpoint
            and save_backend == "dcp"
            and async_mode == "disabled"
            and self.ft_manager is None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            # DCP planning exchanges Python objects. If no process group is
            # supplied, those object collectives use the default NCCL group and
            # stage their payloads on CUDA. Keep synchronous checkpoint
            # coordination on CPU, matching TorchTitan's async DCP path.
            logger.info(
                "Creating a dedicated Gloo process group for synchronous "
                "DCP coordination."
            )
            self._sync_dcp_process_group = dist.new_group(backend="gloo")

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        if force:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    @staticmethod
    def _dist_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _dist_world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _export_dtype(self) -> torch.dtype | None:
        export_dtype = getattr(self.job_config.checkpoint, "export_dtype", None)
        if export_dtype == "float16":
            return torch.float16
        if export_dtype == "bfloat16":
            return torch.bfloat16
        if export_dtype == "float32":
            return torch.float32
        return None

    @staticmethod
    def _model_parts_from_states(states: dict[str, Any]) -> list[torch.nn.Module]:
        model_wrapper = states.get("model")
        model_parts = getattr(model_wrapper, "model", None)
        if model_parts:
            return list(model_parts)
        if isinstance(model_wrapper, torch.nn.Module):
            return [model_wrapper]
        if isinstance(model_wrapper, (list, tuple)):
            return [part for part in model_wrapper if isinstance(part, torch.nn.Module)]
        raise RuntimeError("Could not find model parts for rank0_model_only checkpoint save.")

    def _rank0_tensor(self, value: Any, rank: int, export_dtype: torch.dtype | None) -> Any:
        if not isinstance(value, torch.Tensor):
            return value if rank == 0 else None

        full_value = value.full_tensor() if hasattr(value, "full_tensor") else value
        if rank != 0:
            return None

        full_value = full_value.detach()
        if export_dtype is not None and full_value.is_floating_point():
            full_value = full_value.to(dtype=export_dtype)
        return full_value.cpu()

    def _manual_rank0_model_state_dict(
        self,
        model: torch.nn.Module,
        rank: int,
        export_dtype: torch.dtype | None,
    ) -> dict[str, Any] | None:
        state_dict = model.state_dict()
        if rank == 0:
            output = {}
        else:
            output = None

        for name, value in state_dict.items():
            rank0_value = self._rank0_tensor(value, rank, export_dtype)
            if rank == 0:
                output[name] = rank0_value
            del rank0_value
        return output

    def _rank0_model_state_dict(
        self,
        model: torch.nn.Module,
        rank: int,
        export_dtype: torch.dtype | None,
    ) -> dict[str, Any] | None:
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if rank != 0:
                return None
            if export_dtype is None:
                return state_dict
            return {
                name: (
                    value.detach().to(dtype=export_dtype).cpu()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )
                for name, value in state_dict.items()
            }
        except Exception as exc:  # pragma: no cover - version-dependent fallback
            logger.warning(
                "get_model_state_dict failed for rank0_model_only checkpoint; "
                "falling back to manual DTensor full_tensor export: %s",
                exc,
            )
            return self._manual_rank0_model_state_dict(model, rank, export_dtype)

    @staticmethod
    def _torch_load_cpu(path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch without weights_only
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _rank0_model_only_step(checkpoint_id: str) -> int | None:
        model_path = os.path.join(checkpoint_id, "model.pt")
        metadata_path = os.path.join(checkpoint_id, "metadata.json")
        if not os.path.isfile(model_path):
            return None
        if not os.path.isfile(metadata_path):
            return 0
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:  # pragma: no cover - best-effort diagnostic path
            logger.warning("Could not read rank0 model-only metadata from %s: %s", metadata_path, exc)
            return 0
        if metadata.get("format") != "rank0_model_only":
            return None
        return int(metadata.get("step") or 0)

    @staticmethod
    def _state_dict_options() -> Any:
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        try:
            return StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                strict=False,
            )
        except TypeError:  # pragma: no cover - version-dependent API
            return StateDictOptions(full_state_dict=True, cpu_offload=True)

    @staticmethod
    def _set_full_model_state_dict(
        model: torch.nn.Module,
        model_state_dict: dict[str, Any],
    ) -> None:
        try:
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            options = CheckpointManager._state_dict_options()
            try:
                set_model_state_dict(
                    model,
                    model_state_dict=model_state_dict,
                    options=options,
                )
            except TypeError:  # pragma: no cover - version-dependent API
                set_model_state_dict(model, model_state_dict, options=options)
            return
        except Exception as exc:  # pragma: no cover - fallback for torch variants
            logger.warning(
                "set_model_state_dict failed for rank0_model_only load; "
                "falling back to module.load_state_dict(strict=False): %s",
                exc,
            )

        incompatible = model.load_state_dict(model_state_dict, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing:
            logger.warning("rank0_model_only load skipped missing keys: %s", missing[:20])
        if unexpected:
            logger.warning("rank0_model_only load skipped unexpected keys: %s", unexpected[:20])

    def _fast_forward_lr_schedulers(self, step: int) -> None:
        schedulers = self.states.get("lr_schedulers") or getattr(self, "lr_schedulers", None)
        if not schedulers or not hasattr(schedulers, "step"):
            logger.warning(
                "rank0_model_only resume could not fast-forward LR schedulers to step %s; "
                "optimizer/scheduler state is not present in model-only checkpoints.",
                step,
            )
            return
        for _ in range(step):
            schedulers.step()
        logger.info("Fast-forwarded LR schedulers to step %s for rank0_model_only resume.", step)

    @torch.no_grad()
    def _load_rank0_model_only(self, checkpoint_id: str, step: int) -> None:
        model_path = os.path.join(checkpoint_id, "model.pt")
        model_parts = self._model_parts_from_states(self.states)
        if len(model_parts) != 1:
            raise NotImplementedError(
                "rank0_model_only checkpoint load currently supports exactly one "
                f"model part, got {len(model_parts)}."
            )

        logger.info(
            "Loading rank0 model-only checkpoint from %s at step %s. "
            "This restores model weights and train_state.step only; optimizer, "
            "scheduler, and dataloader state are not restored.",
            model_path,
            step,
        )
        payload = self._torch_load_cpu(model_path)
        model_state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(model_state_dict, dict):
            raise RuntimeError(f"Invalid rank0 model-only checkpoint payload at {model_path}.")

        self._set_full_model_state_dict(model_parts[0], model_state_dict)
        train_state = self.states.get("train_state")
        if train_state is not None and hasattr(train_state, "step"):
            train_state.step = step
            if hasattr(train_state, "skipped_step"):
                train_state.skipped_step = 0
            if hasattr(train_state, "token"):
                train_state.token = 0
            if hasattr(train_state, "global_avg_losses"):
                train_state.global_avg_losses = []
            if hasattr(train_state, "global_max_losses"):
                train_state.global_max_losses = []
            if hasattr(train_state, "log_steps"):
                train_state.log_steps = []
        self._fast_forward_lr_schedulers(step)
        del payload, model_state_dict
        GarbageCollection.collect("GC collection for rank0 model-only checkpoint loading.")

    @torch.no_grad()
    def _save_rank0_model_only(self, curr_step: int, force: bool = False) -> None:
        if not self._should_save(curr_step, force):
            return

        if not getattr(self.job_config.checkpoint, "model_weights_only", False):
            raise ValueError(
                "checkpoint.save_backend=rank0_model_only requires "
                "--checkpoint.model_weights_only because the saved checkpoint "
                "cannot resume optimizer, scheduler, dataloader, or train_state."
            )

        rank = self._dist_rank()
        checkpoint_id = self._create_checkpoint_id(curr_step)
        export_dtype = self._export_dtype()
        model_parts = self._model_parts_from_states(self.states)
        if len(model_parts) != 1:
            raise NotImplementedError(
                "rank0_model_only checkpoint save currently supports exactly one "
                f"model part, got {len(model_parts)}."
            )

        logger.info(
            "Saving rank0 model-only checkpoint to %s at step %s "
            "(export_dtype=%s, world_size=%s).",
            checkpoint_id,
            curr_step,
            getattr(self.job_config.checkpoint, "export_dtype", None),
            self._dist_world_size(),
        )

        begin = time.monotonic()
        self._barrier()
        state_dict = self._rank0_model_state_dict(model_parts[0], rank, export_dtype)

        if rank == 0:
            os.makedirs(checkpoint_id, exist_ok=True)
            tmp_path = os.path.join(checkpoint_id, "model.pt.tmp")
            model_path = os.path.join(checkpoint_id, "model.pt")
            metadata_path = os.path.join(checkpoint_id, "metadata.json")
            torch.save({"model": state_dict, "step": curr_step}, tmp_path)
            os.replace(tmp_path, model_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "format": "rank0_model_only",
                        "step": curr_step,
                        "world_size": self._dist_world_size(),
                        "export_dtype": getattr(
                            self.job_config.checkpoint, "export_dtype", None
                        ),
                        "model_weights_only": True,
                        "resume_semantics": "model_weights_only_not_exact_training_resume",
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")

        del state_dict
        GarbageCollection.collect("GC collection for rank0 model-only checkpoint save.")
        self._barrier()
        logger.info(
            "Finished saving rank0 model-only checkpoint in %.2f seconds.",
            time.monotonic() - begin,
        )

    def _dcp_save_with_gc(self, states: dict[str, Any], checkpoint_id: str) -> None:
        if self._sync_dcp_process_group is None:
            raise RuntimeError(
                "Synchronous distributed DCP save requires its Gloo process group."
            )
        dcp.save(
            states,
            checkpoint_id=checkpoint_id,
            process_group=self._sync_dcp_process_group,
        )
        GarbageCollection.collect("GC collection invoked by checkpointer.")

    def _save_last_step_with_gloo(self, curr_step: int) -> None:
        # Preserve TorchTitan's final-checkpoint behavior while routing DCP's
        # planning collectives through the dedicated CPU process group.
        if self.last_save_model_weights_only:
            self.states = self._states_to_load(model_only=True)
            if self.export_dtype != torch.float32:
                self.states = {
                    key: value.to(self.export_dtype)
                    for key, value in self.states.items()
                }
            logger.info(
                "Saving a model weights only checkpoint in %s at last step, step %s.",
                self.export_dtype,
                curr_step,
            )
        else:
            logger.info("Saving a full checkpoint at last step, step %s.", curr_step)

        self._dcp_save_with_gc(
            self.states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
        )

    def _save_sync_dcp_with_gloo(self, curr_step: int, force: bool = False) -> None:
        if not self._should_save(curr_step, force):
            return

        begin = time.monotonic()
        logger.info("Saving the checkpoint with Gloo DCP coordination.")
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._async_wait()
        if force:
            self._save_last_step_with_gloo(curr_step)
        else:
            self._dcp_save_with_gc(self.states, checkpoint_id=checkpoint_id)
        self._purge_stale_checkpoints()
        logger.info(
            "Finished saving the checkpoint with Gloo DCP coordination "
            "in %.2f seconds.",
            time.monotonic() - begin,
        )

    @torch.no_grad()
    def save(self, curr_step: int, force: bool = False) -> None:
        if self._should_save(curr_step, force):
            # Checkpoint collectives may need CUDA memory outside PyTorch's
            # allocator. Collect unreachable tensor cycles first, then return
            # all unoccupied cached blocks on every rank before saving.
            GarbageCollection.collect("GC collection before checkpoint save.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        save_backend = getattr(self.job_config.checkpoint, "save_backend", "dcp")
        if save_backend == "rank0_model_only":
            self._save_rank0_model_only(curr_step=curr_step, force=force)
            return
        if self._sync_dcp_process_group is not None:
            self._save_sync_dcp_with_gloo(curr_step=curr_step, force=force)
            return
        super().save(curr_step=curr_step, force=force)

    @staticmethod
    def _uses_wrapped_model_namespace(checkpoint_id: str, states: dict[str, Any]) -> bool:
        """Detect DCP checkpoints saved as {"model": model.state_dict()}.

        TorchTitan's model-only initial load expects a flat model state dict, while
        our HF conversion utility historically saved the flat state dict under a
        top-level "model" namespace. Detect that format so pretrained converted
        checkpoints actually hydrate the model instead of being skipped by partial
        load.
        """

        sample_key = next(iter(states), None)
        if sample_key is None:
            return False

        try:
            metadata = dcp.filesystem.FileSystemReader(checkpoint_id).read_metadata()
        except Exception as exc:  # pragma: no cover - best-effort diagnostic path
            logger.warning(
                "Could not read DCP metadata from %s to detect model namespace: %s",
                checkpoint_id,
                exc,
            )
            return False

        checkpoint_keys = metadata.state_dict_metadata.keys()
        return sample_key not in checkpoint_keys and f"model.{sample_key}" in checkpoint_keys

    @staticmethod
    def _checkpoint_state_dict_keys(checkpoint_id: str) -> set[str] | None:
        try:
            metadata = dcp.filesystem.FileSystemReader(checkpoint_id).read_metadata()
        except Exception as exc:  # pragma: no cover - best-effort diagnostic path
            logger.warning(
                "Could not read DCP metadata from %s for post-load hooks: %s",
                checkpoint_id,
                exc,
            )
            return None
        return set(metadata.state_dict_metadata.keys())

    def _run_model_only_post_load_hooks(self, checkpoint_keys: set[str] | None) -> None:
        model_wrapper = self.states.get("model")
        model_parts = getattr(model_wrapper, "model", None)
        if not model_parts:
            return

        hooked = []
        seen = set()
        for model_part in model_parts:
            if not hasattr(model_part, "modules"):
                continue
            for module in model_part.modules():
                hook = getattr(module, "post_initial_model_weight_load", None)
                if not callable(hook) or id(module) in seen:
                    continue
                seen.add(id(module))
                if hook(checkpoint_keys=checkpoint_keys):
                    hooked.append(module.__class__.__name__)

        if hooked:
            logger.info(
                "Ran post-initial model weight load hooks for: %s",
                ", ".join(hooked),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load checkpoint with ``allow_partial_load=True`` so that new
        parameters absent from the checkpoint are silently skipped instead
        of raising ``RuntimeError: Missing key …``."""

        if self.ft_manager:
            self._ft_load()

        if not self.enable_checkpoint:
            return False

        model_only = False
        if not os.path.exists(self.folder):
            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "initial_load_full_checkpoint is specified but the path is not valid."
                    )
                model_only = self.initial_load_model_weights_only
            else:
                return False
        else:
            if self.initial_load_path:
                logger.info(
                    "`initial_load_path` is provided but the checkpoint folder exists. "
                    "Checkpointer will use the checkpoints from the checkpoint folder."
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                return False

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        rank0_model_only_step = self._rank0_model_only_step(checkpoint_id)
        if rank0_model_only_step is not None:
            self._load_rank0_model_only(
                checkpoint_id=checkpoint_id,
                step=rank0_model_only_step or step,
            )
            logger.info(
                "Finished loading the rank0 model-only checkpoint in %.2f seconds.",
                time.monotonic() - begin,
            )
            return True

        states = self._states_to_load(model_only)
        if model_only and self._uses_wrapped_model_namespace(checkpoint_id, states):
            logger.info(
                "Detected model-prefixed model-only checkpoint; loading initial "
                "weights through top-level 'model' namespace."
            )
            states = {"model": states}
        dcp.load(
            states,
            checkpoint_id=checkpoint_id,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        if model_only:
            self._run_model_only_post_load_hooks(
                self._checkpoint_state_dict_keys(checkpoint_id),
            )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True
