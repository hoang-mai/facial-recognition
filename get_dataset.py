import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# Define the Dataset class
class MaiHoang(Dataset):
    def __init__(self, img_dir, csv_file):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, str(self.labels.iloc[idx, 1]), str(self.labels.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')  # Ensure image is in RGB format
        label = self.labels.iloc[idx, 1] - 1  # Assuming labels are 1-based and need to be 0-based

        if self.transform:
            image = self.transform(image)

        return image, label
