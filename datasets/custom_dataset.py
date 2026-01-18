"""
Custom Dataset Loader for Anomaly Detection
============================================

Supports the following folder structure:
dataset_root/
├── train/
│   └── good/
└── test/
    ├── good/
    └── NOK/ (or defect types directly)
        ├── bubble/
        ├── scratch/
        └── ...
"""

import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for Anomaly Detection.

    Supports flexible folder structures with good/defect images.
    Images are resized to a fixed size and normalized using ImageNet statistics.
    """

    def __init__(
        self,
        source,
        resize=512,
        imagesize=512,
        split=DatasetSplit.TRAIN,
        **kwargs,
    ):
        """
        Initialize dataset.

        Args:
            source: Root directory containing train/test folders
            resize: Resize parameter (deprecated, use imagesize)
            imagesize: Target image size (square)
            split: Dataset split (TRAIN, VAL, or TEST)
        """
        super().__init__()
        self.source = source
        self.split = split

        # Setup transforms
        # NOTE: Resize to exact size without cropping to avoid cutting off image parts
        self.transform_img = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),  # Direct resize to target size
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),  # Direct resize to target size
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)

        # Load image data
        self.data_to_iterate = self.get_image_data()

    def get_image_data(self):
        """
        Scan dataset directory and collect image paths.

        Expected structure:
        source/
        ├── train/
        │   └── good/
        └── test/
            ├── good/
            └── NOK/  (or defect types directly)

        Returns:
            List of [anomaly_type, image_path, mask_path] entries
        """
        data_to_iterate = []

        split_path = os.path.join(self.source, self.split.value)

        if not os.path.exists(split_path):
            raise ValueError(f"Split path does not exist: {split_path}")

        # List all subdirectories (good, NOK, or defect types)
        subdirs = sorted(os.listdir(split_path))

        for subdir in subdirs:
            subdir_path = os.path.join(split_path, subdir)

            if not os.path.isdir(subdir_path):
                continue

            # Check if this is "NOK" directory
            if subdir.upper() == "NOK":
                # Check if NOK has subdirectories (defect types) or images directly
                nok_contents = sorted(os.listdir(subdir_path))
                has_subdirs = any(os.path.isdir(os.path.join(subdir_path, item)) for item in nok_contents)

                if has_subdirs:
                    # Nested defect types (e.g., NOK/bubble/, NOK/scratch/)
                    for defect_type in nok_contents:
                        defect_path = os.path.join(subdir_path, defect_type)
                        if not os.path.isdir(defect_path):
                            continue

                        # Get all images in this defect type
                        images = sorted([
                            f for f in os.listdir(defect_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                        ])

                        for img_name in images:
                            img_path = os.path.join(defect_path, img_name)
                            # [anomaly_type, image_path, mask_path]
                            data_to_iterate.append([defect_type, img_path, None])
                else:
                    # Images directly in NOK/ folder
                    images = sorted([
                        f for f in nok_contents
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ])

                    for img_name in images:
                        img_path = os.path.join(subdir_path, img_name)
                        # [anomaly_type, image_path, mask_path]
                        data_to_iterate.append(["NOK", img_path, None])

            else:
                # Direct defect type folder (e.g., "good")
                anomaly_type = "good" if subdir.lower() == "good" else subdir

                images = sorted([
                    f for f in os.listdir(subdir_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ])

                for img_name in images:
                    img_path = os.path.join(subdir_path, img_name)
                    data_to_iterate.append([anomaly_type, img_path, None])

        print(f"Loaded {len(data_to_iterate)} images from {split_path}")

        return data_to_iterate

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: Item index

        Returns:
            Dictionary with image, mask, is_anomaly, and image_path
        """
        anomaly_type, image_path, mask_path = self.data_to_iterate[idx]

        # Load and transform image
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # Load mask if available
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            # Create zero mask for normal images
            mask = torch.zeros([1, *image.size()[1:]])

        # Determine if anomalous
        is_anomaly = int(anomaly_type.lower() != "good")

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": is_anomaly,
            "image_path": image_path,
        }

    def __len__(self):
        """Return dataset size."""
        return len(self.data_to_iterate)
