import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TRAIN_IMG_SIZE,
    RANDOM_CROP_SCALE, RANDOM_CROP_RATIO, ROTATION_DEGREES,
    COLOR_JITTER, MIXUP_ALPHA
)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BinaryCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["fresh", "rotten"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                # Determine if fresh or rotten based on class name
                is_rotten = "rotten" in class_name.lower()
                binary_label = 1 if is_rotten else 0
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(binary_label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_transforms(is_training=True):
    """Create transforms for training or testing"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                TRAIN_IMG_SIZE,
                scale=RANDOM_CROP_SCALE,
                ratio=RANDOM_CROP_RATIO,
                antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(ROTATION_DEGREES),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir, batch_size=BATCH_SIZE, binary=False):
    """
    Get dataloaders for either multi-class or binary classification
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for dataloaders
        binary: If True, use binary classification (fresh vs rotten)
    """
    # Create transforms
    train_transform = create_transforms(is_training=True)
    test_transform = create_transforms(is_training=False)
    
    # Select dataset class based on classification type
    dataset_class = BinaryCustomDataset if binary else CustomDataset
    
    # Create datasets
    train_dataset = dataset_class(
        os.path.join(data_dir, "Train"),
        transform=train_transform
    )
    test_dataset = dataset_class(
        os.path.join(data_dir, "Test"),
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, test_loader, len(train_dataset.classes)

# Add this to your data_processing.py
def standardize_class_names(class_name):
    # Map misspelled classes to their correct versions
    name_mapping = {
        "freshpatato": "freshpotato",
        "freshtamto": "freshtomato",
        "rottenpatato": "rottenpotato",
        "rottentamto": "rottentomato"
    }
    return name_mapping.get(class_name, class_name)

# Then modify your CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.raw_classes = sorted(os.listdir(root_dir))
        self.classes = sorted(set(standardize_class_names(cls) for cls in self.raw_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for raw_class_name in self.raw_classes:
            class_dir = os.path.join(root_dir, raw_class_name)
            if os.path.isdir(class_dir):
                standard_class_name = standardize_class_names(raw_class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[standard_class_name])                        