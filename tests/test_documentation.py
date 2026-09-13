import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from tools.check_configs import CATALOG, ROOT, inspect_configs, render_catalog


def documentation_paths():
    return sorted(set(ROOT.glob("*.md")) | set((ROOT / "docs").glob("*.md")) | {
        ROOT / "uttt_nvs/README.md", ROOT / "uttt_llm/README.md", ROOT / "uttt_nvs/data/README.md",
    })


def test_documentation_local_links_resolve():
    errors = []
    for path in documentation_paths():
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text, f"Replacement character in {path}"
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for destination in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
            parsed = urlsplit(destination)
            if parsed.scheme or not parsed.path:
                continue
            target = path.parent / unquote(parsed.path)
            if not target.exists():
                errors.append(f"{path.relative_to(ROOT)} -> {destination}")
    assert not errors, "\n".join(errors)


def test_documentation_json_examples_parse():
    for path in documentation_paths():
        for example in re.findall(r"```json\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL):
            json.loads(example)


def test_configuration_catalog_is_current():
    entries, errors = inspect_configs()
    assert not errors
    assert CATALOG.read_text(encoding="utf-8") == render_catalog(entries)


def test_static_checker_does_not_import_gpu_stack():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from tools.check_configs import inspect_configs; "
        "entries, errors = inspect_configs(); assert not errors; "
        "assert not {'torch', 'triton', 'flash_attn'} & set(sys.modules)",
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
