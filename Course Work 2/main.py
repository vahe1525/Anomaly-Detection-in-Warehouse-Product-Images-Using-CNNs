import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        # Halves LR when val_loss doesn't improve for 3 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)

    def fit(self, train_loader, val_loader, epochs=30, patience=5):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

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
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    break

        # Restore the best weights found during training
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model weights (val_loss={best_val_loss:.4f})")

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
    torch.manual_seed(42)

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
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # WeightedRandomSampler — gives anomaly images higher pick probability
    # so each batch sees roughly 50% good / 50% anomaly despite the 260 vs 60 imbalance
    class_counts = train_df['label'].value_counts()          # {0: ~210, 1: ~48}
    class_weights = 1.0 / class_counts                       # minority gets higher weight
    sample_weights = train_df['label'].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True                                     # allows reusing anomaly images
    )
    # note: sampler and shuffle=True are mutually exclusive — sampler handles ordering
    train_loader = DataLoader(WoodAnomalyDataset(train_df, train_transform), batch_size=32, sampler=sampler, shuffle = False)
    val_loader = DataLoader(WoodAnomalyDataset(val_df, eval_transform), batch_size=32)
    test_loader = DataLoader(WoodAnomalyDataset(test_df, eval_transform), batch_size=32)

    # 4. Training and Evaluation
    engine = WoodAnomalyEngine()
    engine.fit(train_loader, val_loader, epochs=30, patience=8)
    test_loss, test_acc = engine.evaluate(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
