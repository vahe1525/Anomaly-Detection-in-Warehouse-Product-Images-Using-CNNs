
import pandas as pd
from sklearn.model_selection import train_test_split
from DataManager import WoodAnomalyDataset
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

def show_batch(images, labels):
    plt.figure(figsize=(12, 8))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()


df = pd.read_csv('wood_metadata.csv')

train_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    stratify=df['label'], 
    random_state=42
)

train_df, val_df = train_test_split(
    train_df, 
    test_size=0.10, 
    stratify=train_df['label'], 
    random_state=42
)

print(f" Train Images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images: {len(test_df)}")

# Ստուգենք անոմալիաների բաշխումը Test-ում
print("\nAnomaly count in Test set.")
print(test_df['label'].value_counts())



data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = WoodAnomalyDataset(train_df, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Ստուգում ենք՝ աշխատում է, թե չէ
images, labels = next(iter(train_loader))
print(f"Batch size: {images.shape}")
show_batch(images, labels)