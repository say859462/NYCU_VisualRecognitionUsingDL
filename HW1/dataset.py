"""Dataset utilities for NYCU VRDL Homework 1."""

import os
from PIL import Image
from torch.utils.data import Dataset


# Define Custom Dataset
class ImageDataset(Dataset):
    """Custom dataset wrapper for train / val / test image folders."""

    def __init__(
        self, root_dir: str = "./Dataset/data", split: str = "train", transform=None
    ):
        """Initialize the dataset from the homework directory structure.

        Args:
            root_dir: Root directory that stores the dataset splits.
            split: One of "train", "val", or "test".
            transform: Optional torchvision transform pipeline.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)

        # Parsing the data according to the requirement
        if self.split in ["train", "val"]:
            # Traverse the directories from label 0 to 99
            for label in range(100):
                class_dir = os.path.join(split_dir, str(label))

                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        self.image_paths.append(
                            os.path.join(class_dir, img_name))
                        self.labels.append(label)

        # Handling test dataset
        elif self.split == "test":
            if os.path.exists(split_dir):
                for img_name in os.listdir(split_dir):
                    self.image_paths.append(os.path.join(split_dir, img_name))
                    # Test dataset do not contain label, using -1 instead here
                    self.labels.append(-1)
        else:
            raise ValueError(
                "split parameter should be 'train', 'val' or 'test'")
            
        self.targets = self.labels
        
    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Load one image-label pair from disk."""
        img_path = self.image_paths[idx]

        # Read image
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Data augmentation if needed
        if self.transform:
            image = self.transform(image)

        return image, label


# Function : Download dataset from given google drive url
def download_dataset(output_dir: str = "./Dataset"):
    """Download and extract the official homework dataset."""
    import gdown
    import tarfile

    url = "https://drive.google.com/file/d/1vxiXJHUo6ZPGxBGXwrsSutOpqfJ6HN9D/view"

    filename = "cv_hw1_data.tar"
    output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{filename}"

    # Check whether the dataset is already downloaded
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        print("Dataset is downloaded successfully.")

    print("Extrating....")

    # Extract the dataset.tar file
    try:
        with tarfile.open(output_path, "r") as tar:
            tar.extractall(path=output_dir)

        print(
            f'All datas has been extracted successfully to directory "{output_dir}" . '
        )

    except tarfile.TarError as e:
        print(f"An error occurred during extraction: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")


if __name__ == "__main__":
    download_dataset()
