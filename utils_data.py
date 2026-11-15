
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from utils import CustomPreprocessing

############# define dataset and dataloader ############

# Custom Dataset
class RetinalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class InferenceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Data Preparation
def prepare_data(csv_path, val_size=0.25):
    df = pd.read_csv(csv_path, low_memory=False)
    # df = df.head(500)
    class_counts = df['label'].value_counts().sort_index()
    print(class_counts)
    # Inverse frequency
    class_weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df['label'],)# random_state=42)    
    print("===> tain val size: ", len(train_df), len(val_df))

    return train_df, val_df, class_counts, class_weights

def get_transforms(image_size):
    ############ define transforms ############
    # Transforms
    train_transforms = transforms.Compose([
        #CustomPreprocessing(),
        transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),  # Resize slightly larger
        transforms.RandomCrop((image_size, image_size)),  # Then random crop to final size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transforms
    inference_transform = transforms.Compose([
        #CustomPreprocessing(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

    return train_transforms, val_test_transforms, inference_transform
