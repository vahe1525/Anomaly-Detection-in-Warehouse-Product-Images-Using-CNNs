import torch
import torch.nn as nn

class WoodAnomalyDetector(nn.Module):
    def __init__(self):
        super(WoodAnomalyDetector, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1), # 3->16 
            nn.BatchNorm2d(16),                        
            nn.ReLU(),                                 
            nn.MaxPool2d(kernel_size=2, stride=2) # 224x224 -> 112x112     
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  #16->32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)                            # 112x112 -> 56x56
        )

        # Վերջնական որոշման մասը (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),                              #Linearing
            nn.Linear(32 * 56 * 56, 128),             
            nn.ReLU(),
            nn.Linear(128, 1),                        
            nn.Sigmoid()                               
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x