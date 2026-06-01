import torch.nn as nn


class WoodAnomalyDetector(nn.Module):
    """
    Custom CNN trained from scratch for binary wood anomaly detection.

    Architecture (input: 3 × 224 × 224 RGB image):
        Block 1: Conv(3→16)  + BatchNorm + ReLU + MaxPool  →  16 × 112 × 112
        Block 2: Conv(16→32) + BatchNorm + ReLU + MaxPool  →  32 ×  56 ×  56
        Block 3: Conv(32→64) + BatchNorm + ReLU + MaxPool  →  64 ×  28 ×  28
        Classifier: Flatten → Linear(50176→128) → ReLU → Dropout(0.5) → Linear(128→1) → Sigmoid
    Output: single probability in [0, 1]  — 0 = Good, 1 = Anomaly
    """

    def __init__(self):
        super(WoodAnomalyDetector, self).__init__()

        # Each block doubles the number of feature maps while halving spatial size.
        # More filters = more pattern types detected.
        # Smaller spatial size = each filter covers a larger area of the image.

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 224×224 → 224×224, 3 ch → 16 ch
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 224×224 → 112×112
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 112×112 → 112×112, 16 ch → 32 ch
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 112×112 → 56×56
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 56×56 → 56×56, 32 ch → 64 ch
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 56×56 → 28×28
        )

        # Classifier: takes the flattened feature maps and produces a single output
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # [batch, 64, 28, 28] → [batch, 50176]
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),             # randomly zeros 50% of neurons during training
            nn.Linear(128, 1),
            nn.Sigmoid()                   # output: probability [0.0, 1.0]
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
