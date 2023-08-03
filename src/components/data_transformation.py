import torch
from torchvision import transforms

class DataTransform:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.def_img_size = (512,512) if torch.cuda.is_available() else (128,128)
    
    def resize(self,img_size = (512,512)): # Returns a Transform Module
        trf = transforms.Compose([
           transforms.Resize(img_size),
           transforms.ToTensor(), 
        ])
        return trf

    def normalized_transform(self,resize=False):
        if resize:
            trf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.def_img_size),
                transforms.Normalize(mean=self.mean , std= self.std)
            ])
            
            return trf
        else:
            trf_noresize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean , std= self.std)
            ])
            
            return trf_noresize