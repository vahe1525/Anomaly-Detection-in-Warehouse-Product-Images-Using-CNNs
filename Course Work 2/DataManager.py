import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class WoodAnomalyDataset(Dataset):
    """
    Այս կլասը հանդիսանում է կապող օղակ սկավառակի վրայի ֆայլերի 
    և մեր նեյրոնային ցանցի միջև:
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Այս մեթոդը կանչվում է ամեն անգամ, երբ մոդելին նոր նկար է պետք:
        """
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label