
import pandas as pd
from sklearn.model_selection import train_test_split
from DataManager import WoodAnomalyDataset
from torch.utils.data import DataLoader
from torchvision import transforms



df = pd.read_csv('wood_metadata.csv')

# Նախ բաժանում ենք ընդհանուր դատասեթը 80% Train և 20% Test
train_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    stratify=df['label'], # Սա ամենակարևոր մասն է
    random_state=42
)

train_df, val_df = train_test_split(
    train_df, 
    test_size=0.10, 
    stratify=train_df['label'], 
    random_state=42
)

print(f"Մարզման նկարներ (Train): {len(train_df)}")
print(f"Ստուգման նկարներ (Val): {len(val_df)}")
print(f"Թեստավորման նկարներ (Test): {len(test_df)}")

# Ստուգենք անոմալիաների բաշխումը Test-ում
print("\nԱնոմալիաների քանակը Test set-ում.")
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
print(f"Batch size: {images.shape}") # Պետք է լինի [32, 3, 224, 224]
