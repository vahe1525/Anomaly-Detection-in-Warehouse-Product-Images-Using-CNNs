import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from DataManager import WoodAnomalyDataset
from model import WoodAnomalyDetector
from MetadataPreparer import prepare_wood_metadata
import os

class WoodAnomalyEngine:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = WoodAnomalyDetector().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def fit(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            val_loss, val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(test_loader), correct / total

    def predict(self, image_path, transform):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
        return "Anomaly" if output.item() > 0.5 else "Good"

def main():
    # 1. Preparation
    if not os.path.exists('wood_metadata.csv'):
        df = prepare_wood_metadata('data/wood')
        df.to_csv('wood_metadata.csv', index=False)
    else:
        df = pd.read_csv('wood_metadata.csv')

    # 2. Splitting
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

    # 3. Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(WoodAnomalyDataset(train_df, transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(WoodAnomalyDataset(val_df, transform), batch_size=32)
    test_loader = DataLoader(WoodAnomalyDataset(test_df, transform), batch_size=32)

    # 4. Training and Evaluation
    engine = WoodAnomalyEngine()
    engine.fit(train_loader, val_loader, epochs=7)
    test_loss, test_acc = engine.evaluate(test_loader)
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
