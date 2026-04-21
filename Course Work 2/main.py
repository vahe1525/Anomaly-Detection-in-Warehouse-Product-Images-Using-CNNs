import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from DataManager import WoodAnomalyDataset
from model import WoodAnomalyDetector
from MetadataPreparer import prepare_wood_metadata
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class WoodAnomalyEngine:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = WoodAnomalyDetector().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        # Halves LR when val_loss doesn't improve for 3 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)

    def fit(self, train_loader, val_loader, epochs=30, patience=8):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

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

            train_loss = running_loss / len(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

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

        return history

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

    def get_predictions(self, loader):
        """Collects all true labels and predicted labels for a data loader."""
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                outputs = self.model(images)
                predicted = (outputs > 0.5).float()
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(predicted.cpu().numpy().flatten())
        return np.array(all_labels, dtype=int), np.array(all_preds, dtype=int)

    def predict(self, image_path, transform):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
        return "Anomaly" if output.item() > 0.5 else "Good"

    def plot_training_curves(self, history):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='o', markersize=3)
        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='green', marker='o', markersize=3)
        ax2.set_title('Validation Accuracy per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.show()
        print("Saved: training_curves.png")

    def plot_confusion_matrix(self, loader, class_names=('Good', 'Anomaly')):
        true_labels, pred_labels = self.get_predictions(loader)
        cm = confusion_matrix(true_labels, pred_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix — Test Set')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.show()
        print("Saved: confusion_matrix.png")

    def plot_sample_predictions(self, test_df, eval_transform, n=8):
        sample = test_df.sample(n=n, random_state=42).reset_index(drop=True)
        cols = 4
        rows = n // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()

        self.model.eval()
        for i, row in sample.iterrows():
            image = Image.open(row['image_path']).convert('RGB')
            tensor = eval_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
            pred_label = int(output.item() > 0.5)
            true_label = int(row['label'])
            pred_name = 'Anomaly' if pred_label == 1 else 'Good'
            true_name = 'Anomaly' if true_label == 1 else 'Good'
            correct = pred_label == true_label

            axes[i].imshow(image.resize((224, 224)))
            axes[i].set_title(
                f"True: {true_name}\nPred: {pred_name}",
                color='green' if correct else 'red',
                fontsize=9,
                fontweight='bold'
            )
            axes[i].axis('off')

        plt.suptitle('Sample Test Predictions  |  green = correct   red = wrong', fontsize=12)
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=150)
        plt.show()
        print("Saved: sample_predictions.png")


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
    class_counts = train_df['label'].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = train_df['label'].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(WoodAnomalyDataset(train_df, train_transform), batch_size=32, sampler=sampler, shuffle=False)
    val_loader = DataLoader(WoodAnomalyDataset(val_df, eval_transform), batch_size=32)
    test_loader = DataLoader(WoodAnomalyDataset(test_df, eval_transform), batch_size=32)

    # 4. Training
    engine = WoodAnomalyEngine()
    history = engine.fit(train_loader, val_loader, epochs=30, patience=8)

    # 5. Test metrics
    test_loss, test_acc = engine.evaluate(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

    true_labels, pred_labels = engine.get_predictions(test_loader)
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['Good', 'Anomaly']))

    # 6. Visualizations
    engine.plot_training_curves(history)
    engine.plot_confusion_matrix(test_loader)
    engine.plot_sample_predictions(test_df, eval_transform, n=8)


if __name__ == '__main__':
    main()
