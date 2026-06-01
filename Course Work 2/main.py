"""
Wood Anomaly Detection — Unified Training Script
=================================================
Switch between the two models by changing MODEL_TYPE below:

    MODEL_TYPE = 'cnn'    -> Custom 3-block CNN trained from scratch
    MODEL_TYPE = 'resnet' -> ResNet18 with ImageNet pretrained weights (transfer learning)

Run:  python main.py
"""

# ── Choose your model here ────────────────────────────────────────────────────
MODEL_TYPE = 'resnet'   # 'cnn'  or  'resnet'
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves PNGs without opening windows
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
from model import WoodAnomalyDetector           # custom CNN
from resnet_model import WoodResNetDetector     # ResNet18 transfer learning


# ─────────────────────────────────────────────────────────────────────────────
#  Engine — all training, evaluation and visualization logic lives here.
#  It works for both models — the only difference is which model gets created
#  in __init__, controlled by MODEL_TYPE.
# ─────────────────────────────────────────────────────────────────────────────

class WoodAnomalyEngine:

    def __init__(self, model_type='cnn', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device     = device
        self.model_type = model_type
        print(f"Device : {self.device}")
        print(f"Model  : {model_type.upper()}")
        print()

        # ── Select model and matching learning rate ──
        if model_type == 'resnet':
            # ResNet18 pretrained on ImageNet — backbone weights are already good,
            # so we use a small LR (0.0001) to update them gently without destroying
            # what they learned from 1.2 million images.
            self.model = WoodResNetDetector(pretrained=True).to(device)
            lr = 0.0001
        else:
            # Custom CNN trained from scratch — weights start random, so we need
            # a larger LR (0.0003) to make meaningful progress per epoch.
            self.model = WoodAnomalyDetector().to(device)
            lr = 0.0003

        self.criterion = nn.BCELoss()
        # Adam optimizer — adapts the learning rate per parameter automatically,
        # making it more efficient than plain SGD on small datasets.
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # LR scheduler — halves the LR when val_loss doesn't improve for 3 epochs.
        # This lets the model converge more precisely once it's near a good solution.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)

    # ── Training loop ──────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, epochs, patience):
        """
        Train the model with early stopping and LR scheduling.

        Each epoch:
          1. Train phase  — forward pass, compute loss, backprop, update weights
          2. Val phase    — evaluate on validation set (no weight updates)
          3. LR scheduler — halve LR if val_loss stagnates for 3 epochs
          4. Early stop   — save best weights; stop if no improvement for `patience` epochs

        Returns history dict (train_loss, val_loss, val_acc per epoch) for plotting.
        """
        best_val_loss    = float('inf')   # any real loss will beat infinity
        patience_counter = 0              # counts epochs with no val_loss improvement
        best_model_state = None           # snapshot of the best weights found so far
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):

            # ── 1. Train phase ────────────────────────────────────────────────
            # model.train() activates Dropout — neurons are randomly zeroed during
            # training to force the network to learn redundant, robust features.
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                # Move batch to GPU or CPU — must match where the model lives
                images = images.to(self.device)
                # labels shape: [batch_size] -> [batch_size, 1]
                # .float() because BCELoss requires float32, not int
                # .unsqueeze(1) adds a dimension to match model output shape [batch, 1]
                labels = labels.float().unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()               # clear gradients from previous batch
                outputs = self.model(images)              # forward pass -> probabilities [0,1]
                loss    = self.criterion(outputs, labels) # BCE: how wrong are we?
                loss.backward()                          # backprop: compute gradient for every weight
                self.optimizer.step()                    # Adam updates every weight
                running_loss += loss.item()

            # ── 2. Validation phase ───────────────────────────────────────────
            train_loss        = running_loss / len(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            current_lr        = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # ── 3. LR scheduler ───────────────────────────────────────────────
            # Pass val_loss to the scheduler. If it hasn't improved for 3 epochs,
            # the scheduler will halve the LR automatically on the next call.
            self.scheduler.step(val_loss)

            # ── 4. Early stopping ─────────────────────────────────────────────
            if val_loss < best_val_loss:
                # New best — save a deep copy of all model weights right now
                best_val_loss    = val_loss
                patience_counter = 0
                # .clone() is critical: without it, best_model_state would be a
                # reference to the same tensors that keep changing each epoch
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}. "
                          f"Best val loss: {best_val_loss:.4f}")
                    break

        # Restore the best weights — not the last epoch's weights, which may have overfit
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model weights (val_loss={best_val_loss:.4f})")

        return history

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, loader):
        """
        Compute average loss and accuracy over any DataLoader.

        model.eval() disables Dropout — during evaluation we want the full
        network active, not random neurons zeroed out.
        torch.no_grad() turns off gradient tracking — we don't need gradients
        during evaluation, so this saves memory and speeds things up.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total   = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs      = self.model(images)
                loss         = self.criterion(outputs, labels)
                running_loss += loss.item()

                # Convert probability to hard prediction: >0.5 -> 1 (Anomaly), else 0 (Good)
                predicted = (outputs > 0.5).float()
                total    += labels.size(0)
                correct  += (predicted == labels).sum().item()

        return running_loss / len(loader), correct / total

    def get_predictions(self, loader):
        """
        Collect all ground-truth labels and model predictions for a full DataLoader.
        Used by the classification report and confusion matrix.
        Returns two numpy int arrays: (true_labels, pred_labels)
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

                # .cpu() moves tensors from GPU back to CPU before converting to numpy
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(predicted.cpu().numpy().flatten())

        return np.array(all_labels, dtype=int), np.array(all_preds, dtype=int)

    # ── Visualizations ─────────────────────────────────────────────────────────

    def plot_training_curves(self, history):
        """
        Two side-by-side plots:
          Left:  training loss and validation loss per epoch
          Right: validation accuracy per epoch
        Shows how the model learned (or overfit) over time.
        """
        prefix = self.model_type   # 'cnn' or 'resnet' — used in title and filename
        epochs = range(1, len(history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, history['val_loss'],   label='Val Loss',   marker='o', markersize=3)
        ax1.set_title(f'{prefix.upper()} — Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='green', marker='o', markersize=3)
        ax2.set_title(f'{prefix.upper()} — Validation Accuracy per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        filename = f'{prefix}_training_curves.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

    def plot_confusion_matrix(self, loader, class_names=('Good', 'Anomaly')):
        """
        Confusion matrix on the test set.

        The 4 cells tell you:
          Top-left     (TN): predicted Good,    actually Good    correct
          Top-right    (FP): predicted Anomaly, actually Good    wrong (false alarm)
          Bottom-left  (FN): predicted Good,    actually Anomaly wrong (missed defect)
          Bottom-right (TP): predicted Anomaly, actually Anomaly correct
        """
        prefix = self.model_type
        true_labels, pred_labels = self.get_predictions(loader)
        cm = confusion_matrix(true_labels, pred_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{prefix.upper()} — Confusion Matrix (Test Set)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        filename = f'{prefix}_confusion_matrix.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

    def plot_sample_predictions(self, test_df, eval_transform, n=8):
        """
        Show n random test images in a grid.
        Title is green if the prediction is correct, red if wrong.
        Good for visually checking what kinds of errors the model makes.
        """
        prefix = self.model_type
        sample = test_df.sample(n=n, random_state=42).reset_index(drop=True)
        cols   = 4
        rows   = n // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()

        self.model.eval()
        for i, row in sample.iterrows():
            image  = Image.open(row['image_path']).convert('RGB')
            # unsqueeze(0) adds a batch dimension: [3,224,224] -> [1,3,224,224]
            # because the model always expects a batch, even for one image
            tensor = eval_transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

            pred_label = int(output.item() > 0.5)
            true_label = int(row['label'])
            pred_name  = 'Anomaly' if pred_label == 1 else 'Good'
            true_name  = 'Anomaly' if true_label == 1 else 'Good'
            correct    = (pred_label == true_label)

            axes[i].imshow(image.resize((224, 224)))
            axes[i].set_title(
                f"True: {true_name}\nPred: {pred_name}",
                color='green' if correct else 'red',
                fontsize=9, fontweight='bold'
            )
            axes[i].axis('off')

        plt.suptitle(f'{prefix.upper()} — Sample Predictions  |  green=correct  red=wrong', fontsize=12)
        plt.tight_layout()
        filename = f'{prefix}_sample_predictions.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main — data pipeline, training, evaluation, visualizations
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)   # fixed seed -> same results every run

    # ── 1. Load or build the metadata CSV ────────────────────────────────────
    # wood_metadata.csv has 3 columns: image_path | parent_folder | label
    # It is built once by scanning data/wood/ and cached so we don't re-scan.
    if not os.path.exists('wood_metadata.csv'):
        df = prepare_wood_metadata('data/wood')
        df.to_csv('wood_metadata.csv', index=False)
    else:
        df = pd.read_csv('wood_metadata.csv')

    print(f"Dataset: {len(df)} total images")
    print(df['label'].value_counts().rename({0: 'Good', 1: 'Anomaly'}).to_string())
    print()

    # ── 2. Split into train / val / test ─────────────────────────────────────
    # Step 1: separate 20% as the test set (never seen during training)
    # Step 2: from the remaining 80%, take 10% as validation
    # Result: ~72% train | ~8% val | 20% test
    #
    # stratify= ensures each split preserves the Good/Anomaly ratio from the full set.
    # Without stratify, a random split might put all anomaly images in one split.
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print()

    # ── 3. Image transforms ───────────────────────────────────────────────────
    # Two separate pipelines — augmentation only on training images.
    # Val and test images are always shown clean so scores are fair and comparable.
    #
    # The ImageNet normalization values (mean/std) are used for BOTH models:
    # - ResNet18 was trained with these exact stats, so it requires them.
    # - The custom CNN also uses them for consistent input scale.

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),   # 50% chance: mirror left<->right
        transforms.RandomVerticalFlip(),     # 50% chance: mirror top<->bottom
        transforms.RandomRotation(15),       # random rotation between -15 and +15 degrees
        transforms.ToTensor(),               # PIL Image -> float tensor, values 0->1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean per channel
                             std=[0.229, 0.224, 0.225])    # ImageNet std per channel
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 4. WeightedRandomSampler — fix class imbalance ────────────────────────
    # Problem: ~209 Good images vs ~48 Anomaly images in train.
    # Without balancing: batches contain ~26 Good + ~6 Anomaly.
    # The model quickly learns "always say Good" -> stuck at ~80% accuracy.
    #
    # Solution: give each Anomaly image a higher pick probability so each batch
    # ends up roughly 50% Good / 50% Anomaly.
    #
    # class_counts:  {0: 209, 1: 48}
    # class_weights: {0: 1/209 ~ 0.005, 1: 1/48 ~ 0.021}  <- anomaly is ~4x more likely
    # sample_weights: one weight per training image based on its label

    class_counts   = train_df['label'].value_counts()
    class_weights  = 1.0 / class_counts
    sample_weights = train_df['label'].map(class_weights).values

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True    # allows the same anomaly image to be picked multiple
                            # times per epoch — necessary since there are only ~48
    )

    # Note: sampler= and shuffle=True are mutually exclusive — sampler handles ordering
    train_loader = DataLoader(WoodAnomalyDataset(train_df, train_transform),
                              batch_size=32, sampler=sampler)
    val_loader   = DataLoader(WoodAnomalyDataset(val_df,   eval_transform), batch_size=32)
    test_loader  = DataLoader(WoodAnomalyDataset(test_df,  eval_transform), batch_size=32)

    # ── 5. Train ───────────────────────────────────────────────────────────────
    # Epochs differ by model:
    # - CNN:    30 epochs max — trains from scratch, needs more time to converge
    # - ResNet: 20 epochs max — pretrained, converges much faster
    epochs   = 20 if MODEL_TYPE == 'resnet' else 30
    patience = 8   # same for both: stop if val_loss doesn't improve for 8 epochs

    engine  = WoodAnomalyEngine(model_type=MODEL_TYPE)
    history = engine.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    # ── 6. Final test evaluation ───────────────────────────────────────────────
    # The test set is only ever used here — after training is fully done.
    # This gives an honest score on data the model has never seen.
    test_loss, test_acc = engine.evaluate(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

    true_labels, pred_labels = engine.get_predictions(test_loader)
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                                target_names=['Good', 'Anomaly']))

    # ── 7. Visualizations ──────────────────────────────────────────────────────
    engine.plot_training_curves(history)
    engine.plot_confusion_matrix(test_loader)
    engine.plot_sample_predictions(test_df, eval_transform, n=8)


if __name__ == '__main__':
    main()
