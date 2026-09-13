import json
import os
from io import BytesIO
import random
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from google.cloud import storage as gcs_storage
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


_S3_CLIENT = None
S3_TRANSFER_CONFIG = TransferConfig(multipart_threshold=5 * 1024**3)  # 5GB

_GCS_CLIENT = None


def build_s3_client():
    global _S3_CLIENT
    S3_CONFIG = Config(region_name="us-west-2", signature_version=UNSIGNED)
    _S3_CLIENT = boto3.client('s3', config=S3_CONFIG)


def destroy_s3_client():
    global _S3_CLIENT
    _S3_CLIENT = None


def build_gcs_client():
    global _GCS_CLIENT
    _GCS_CLIENT = gcs_storage.Client()


def destroy_gcs_client():
    global _GCS_CLIENT
    _GCS_CLIENT = None


def open_s3_file(s3_path):
    file_stream = BytesIO()
    s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)
    _S3_CLIENT.download_fileobj(s3_bucket, s3_key, file_stream, Config=S3_TRANSFER_CONFIG)
    file_stream.seek(0)
    return file_stream


def open_gcs_file(gcs_path):
    bucket_name, blob_name = gcs_path[len("gs://"):].split("/", 1)
    blob = _GCS_CLIENT.bucket(bucket_name).blob(blob_name)
    return BytesIO(blob.download_as_bytes(timeout=60))


def open_file(path):
    """Open a file from local path, S3, or GCS."""
    if path.startswith("s3://"):
        return open_s3_file(path)
    elif path.startswith("gs://"):
        return open_gcs_file(path)
    else:
        return open(path, "rb")



def resize_and_crop(image, target_size, fxfycxcy):
    """
    Resize and crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (height, width) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    original_width, original_height = image.size  # PIL image is (width, height)
    target_height, target_width = target_size
    
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    processed_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop image
    if new_width > target_width or new_height > target_height:
        processed_image = processed_image.crop((left, top, right, bottom))
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale - left
    new_cy = cy * scale - top
    
    return processed_image, [new_fx, new_fy, new_cx, new_cy]



class NVSDataset(Dataset):

    MAX_CONSECUTIVE_FAILURES = 100
    def __init__(self, config):
        """
        image_size is (h, w) or just a int (as size).
        """
        super().__init__()

        self.num_views = config.training.num_views
        self.image_size = (config.model.image_size, config.model.image_size)

        self.use_s3 = config.training.dataset_path.startswith("s3://")
        self.use_gcs = config.training.dataset_path.startswith("gs://")
        if self.use_s3:
            build_s3_client()
        elif self.use_gcs:
            build_gcs_client()
        with open_file(config.training.dataset_path) as f:
            content = f.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            self.all_camera_paths = content.split("\n")
        if self.use_s3:
            destroy_s3_client()     # Destroy to avoid spawn
        elif self.use_gcs:
            destroy_gcs_client()    # Destroy to avoid spawn

        self.all_camera_paths = pd.array(
            [s for s in self.all_camera_paths if len(s) > 0], dtype="string[pyarrow]"
        )

        if config.training.get("max_num_objects", -1) > 0:
            all_dataset_len = len(self.all_camera_paths)
            self.all_camera_paths = self.all_camera_paths[: config.training.max_num_objects]
            self.all_camera_paths = self.all_camera_paths * (all_dataset_len // len(self.all_camera_paths))


    def __len__(self):
        return len(self.all_camera_paths)
    
    def __getitem__(self, index):
        if self.use_s3 and _S3_CLIENT is None:
            build_s3_client()
        elif self.use_gcs and _GCS_CLIENT is None:
            build_gcs_client()

        try:
            data_point_path = os.path.join(self.all_camera_paths[index])
            data_point_base_dir = os.path.dirname(data_point_path)
            with open_file(data_point_path) as f:
                images_info = json.load(f)["frames"]
            
            # If the num_views is larger than the number of images, use all images
            indices = random.sample(range(len(images_info)), self.num_views)
            
            def load_single_view(index):
                info = images_info[index]
                
                fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
                
                w2c = torch.tensor(info["w2c"])
                c2w = torch.inverse(w2c)

                # print(info["file_path"])
                
                # Load image from file_path using PIL and convert to torch tensor
                image_path = os.path.join(data_point_base_dir, info["file_path"])
                with open_file(image_path) as f:
                    image = Image.open(f)
                    image.load()
                
                image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)

                # Convert RGBA to RGB if needed
                if image.mode == 'RGBA':
                    # Create a white background and paste the RGBA image on it
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = rgb_image
                elif image.mode != 'RGB':
                    # Convert any other mode to RGB
                    image = image.convert('RGB')
                
                return c2w, fxfycxcy, transforms.ToTensor()(image)
            
            # Parallel loading using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(load_single_view, indices))
        except Exception as exc:
            self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1
            if self._consecutive_failures > self.MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Failed to load {self.MAX_CONSECUTIVE_FAILURES} samples in a row; "
                    f"the last failing path was '{self.all_camera_paths[index]}'. "
                    f"Check that `dataset_path` points at a readable manifest and that "
                    f"every path inside it is reachable."
                ) from exc
            print(f"[NVSDataset] skipping unreadable sample {index}: {exc}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        self._consecutive_failures = 0
        
        c2w_list, fxfycxcy_list, image_list = zip(*results)
        
        c2ws = torch.stack(c2w_list)

        image_indices = torch.tensor(indices).long().unsqueeze(-1)
        scene_indices = torch.tensor(index).long().unsqueeze(0).expand_as(image_indices)
        data_indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
            "index": data_indices,
        }
