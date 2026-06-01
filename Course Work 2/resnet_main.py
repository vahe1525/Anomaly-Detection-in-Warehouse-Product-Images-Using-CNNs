"""
ResNet18 Transfer Learning — Wood Anomaly Detection
=====================================================
This script trains a ResNet18-based model on the wood dataset using
transfer learning. The backbone (feature extractor) is pretrained on
ImageNet; only the final classification head is new.

Run:  python resnet_main.py
Outputs: resnet_training_curves.png, resnet_confusion_matrix.png,
         resnet_sample_predictions.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from DataManager import WoodAnomalyDataset
from MetadataPreparer import prepare_wood_metadata
from resnet_model import WoodResNetDetector


# ─────────────────────────────────────────────────────────────────────────────
#  Engine — handles training, evaluation, and visualizations
# ─────────────────────────────────────────────────────────────────────────────

class WoodResNetEngine:
    """
    Training and evaluation engine for the ResNet18 transfer learning model.

    Key differences from the scratch CNN engine:
    - Lower learning rate (0.0001 vs 0.0003) — pretrained weights are already
      good, so we update them gently to avoid destroying what they learned.
    - Fewer max epochs (20 vs 30) — transfer learning converges much faster.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        self.model     = WoodResNetDetector(pretrained=True).to(self.device)
        self.criterion = nn.BCELoss()

        # Lower LR than the scratch model because the backbone is already pretrained
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        # Halve the LR when val_loss stops improving for 3 consecutive epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)

    # ── Training loop ──────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, epochs=20, patience=8):
        """
        Train the model with early stopping and LR scheduling.

        Args:
            train_loader: DataLoader for training set
            val_loader:   DataLoader for validation set
            epochs:       Maximum number of training epochs
            patience:     Early stopping — stop if val_loss doesn't improve
                          for this many consecutive epochs

        Returns:
            history dict with train_loss, val_loss, val_acc per epoch
        """
        best_val_loss    = float('inf')
        patience_counter = 0
        best_model_state = None
        history          = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):

            # ── Train phase: model learns from training batches ──
            self.model.train()  # enables Dropout, BatchNorm in train mode
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)  # shape [B, 1]

                self.optimizer.zero_grad()         # clear gradients from last batch
                outputs = self.model(images)        # forward pass → probabilities
                loss    = self.criterion(outputs, labels)  # compute BCE loss
                loss.backward()                    # backpropagation
                self.optimizer.step()              # update weights
                running_loss += loss.item()

            # ── Validation phase: check performance on unseen data ──
            train_loss          = running_loss / len(train_loader)
            val_loss, val_acc   = self.evaluate(val_loader)
            current_lr          = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # LR scheduler: halve LR if val_loss didn't improve for 3 epochs
            self.scheduler.step(val_loss)

            # ── Early stopping logic ──
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                # Save a deep copy of the best weights found so far
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}. "
                          f"Best val loss: {best_val_loss:.4f}")
                    break

        # Restore the best weights (not the last weights, which may have overfit)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model weights (val_loss={best_val_loss:.4f})")

        return history

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, loader):
        """
        Compute average loss and accuracy on any data loader.
        Returns: (avg_loss, accuracy)
        """
        self.model.eval()  # disables Dropout — full network is used
        running_loss = 0.0
        correct      = 0
        total        = 0

        with torch.no_grad():  # no gradient tracking during evaluation
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs   = self.model(images)
                loss      = self.criterion(outputs, labels)
                running_loss += loss.item()

                predicted  = (outputs > 0.5).float()  # probability → 0 or 1
                total     += labels.size(0)
                correct   += (predicted == labels).sum().item()

        return running_loss / len(loader), correct / total

    def get_predictions(self, loader):
        """
        Collect all true labels and model predictions from a data loader.
        Used for classification report and confusion matrix.

        Returns: (true_labels, pred_labels) as numpy int arrays
        """
        self.model.eval()
        all_labels = []
        all_preds  = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs   = self.model(images)
                predicted = (outputs > 0.5).float()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(predicted.cpu().numpy().flatten())

        return np.array(all_labels, dtype=int), np.array(all_preds, dtype=int)

    # ── Visualizations ─────────────────────────────────────────────────────

    def plot_training_curves(self, history):
        """Plot train/val loss and validation accuracy over epochs. Saves PNG."""
        epochs = range(1, len(history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, history['val_loss'],   label='Val Loss',   marker='o', markersize=3)
        ax1.set_title('ResNet18 — Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='green', marker='o', markersize=3)
        ax2.set_title('ResNet18 — Validation Accuracy per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('resnet_training_curves.png', dpi=150)
        plt.show()
        print("Saved: resnet_training_curves.png")

    def plot_confusion_matrix(self, loader, class_names=('Good', 'Anomaly')):
        """
        Plot confusion matrix on any loader.
        Rows = true labels, Columns = predicted labels.
        """
        true_labels, pred_labels = self.get_predictions(loader)
        cm = confusion_matrix(true_labels, pred_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('ResNet18 — Confusion Matrix (Test Set)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        plt.savefig('resnet_confusion_matrix.png', dpi=150)
        plt.show()
        print("Saved: resnet_confusion_matrix.png")

    def plot_sample_predictions(self, test_df, eval_transform, n=8):
        """
        Show n random test images in a grid with true vs predicted labels.
        Green title = correct prediction, Red title = wrong prediction.
        """
        sample = test_df.sample(n=n, random_state=42).reset_index(drop=True)
        cols   = 4
        rows   = n // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()

        self.model.eval()
        for i, row in sample.iterrows():
            image  = Image.open(row['image_path']).convert('RGB')
            tensor = eval_transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

            pred_label = int(output.item() > 0.5)
            true_label = int(row['label'])
            pred_name  = 'Anomaly' if pred_label == 1 else 'Good'
            true_name  = 'Anomaly' if true_label == 1 else 'Good'
            correct    = pred_label == true_label

            axes[i].imshow(image.resize((224, 224)))
            axes[i].set_title(
                f"True: {true_name}\nPred: {pred_name}",
                color='green' if correct else 'red',
                fontsize=9,
                fontweight='bold'
            )
            axes[i].axis('off')

        plt.suptitle('ResNet18 — Sample Predictions  |  green = correct   red = wrong', fontsize=12)
        plt.tight_layout()
        plt.savefig('resnet_sample_predictions.png', dpi=150)
        plt.show()
        print("Saved: resnet_sample_predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main — data pipeline + train + evaluate + visualize
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)  # reproducible results every run

    # ── 1. Load metadata CSV ───────────────────────────────────────────────
    # wood_metadata.csv maps each image path to its label (0=Good, 1=Anomaly)
    if not os.path.exists('wood_metadata.csv'):
        df = prepare_wood_metadata('data/wood')
        df.to_csv('wood_metadata.csv', index=False)
    else:
        df = pd.read_csv('wood_metadata.csv')

    print(f"Dataset: {len(df)} images total")
    print(df['label'].value_counts().rename({0: 'Good', 1: 'Anomaly'}).to_string())
    print()

    # ── 2. Data splits ─────────────────────────────────────────────────────
    # stratify=label ensures each split keeps the same Good/Anomaly ratio
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42)

    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print()

    # ── 3. Transforms ──────────────────────────────────────────────────────
    # The ImageNet normalization values (mean/std) are required by ResNet18
    # because those are the exact statistics it was trained on.
    # We already use these in the scratch model too — no change needed here.

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),    # random left/right mirror
        transforms.RandomVerticalFlip(),      # random up/down mirror
        transforms.RandomRotation(15),        # random ±15° rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 4. Class balancing with WeightedRandomSampler ──────────────────────
    # Dataset has ~260 Good and ~60 Anomaly images.
    # Without balancing, the model would predict "Good" for everything.
    # WeightedRandomSampler gives each anomaly image a ~4× higher chance
    # of being picked per batch, so batches end up roughly 50/50.
    class_counts    = train_df['label'].value_counts()   # {0: ~209, 1: ~48}
    class_weights   = 1.0 / class_counts                 # minority gets higher weight
    sample_weights  = train_df['label'].map(class_weights).values

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True   # allows reusing anomaly images (only 48 of them)
    )

    train_loader = DataLoader(WoodAnomalyDataset(train_df, train_transform),
                              batch_size=32, sampler=sampler)
    val_loader   = DataLoader(WoodAnomalyDataset(val_df,   eval_transform), batch_size=32)
    test_loader  = DataLoader(WoodAnomalyDataset(test_df,  eval_transform), batch_size=32)

    # ── 5. Train ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Model: ResNet18 Transfer Learning")
    print("  Backbone: pretrained on ImageNet (1.2M images)")
    print("  Task: Binary — Good (0) vs Anomaly (1)")
    print("=" * 60)
    print()

    engine  = WoodResNetEngine()
    history = engine.fit(train_loader, val_loader, epochs=20, patience=8)

    # ── 6. Final test evaluation ───────────────────────────────────────────
    test_loss, test_acc = engine.evaluate(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

    true_labels, pred_labels = engine.get_predictions(test_loader)
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                                target_names=['Good', 'Anomaly']))

    # ── 7. Visualizations ─────────────────────────────────────────────────
    engine.plot_training_curves(history)
    engine.plot_confusion_matrix(test_loader)
    engine.plot_sample_predictions(test_df, eval_transform, n=8)


if __name__ == '__main__':
    main()
