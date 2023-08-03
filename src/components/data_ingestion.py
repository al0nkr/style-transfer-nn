import os
from PIL import Image
import torch
from torchvision import transforms
import data_transformation as DT

class DataIngestion:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = (512,512) if torch.cuda.is_available() else (128,128)
        self.img_transform = DT.DataTransform().resize()
    def load_image(self,image_path):
        img = Image.open(image_path)
        img = self.img_transform(img).unsqueeze(0)
        return img.to(self.device, torch.float)

    def load_content_paths(content_dir):
        content_images = []
        for filename in os.listdir(content_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(content_dir, filename)
                content_images.append(img_path)
        return content_images

    def load_style_paths(style_dir):
        style_images = []
        for filename in os.listdir(style_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(style_dir, filename)
                style_images.append(img_path)
        return style_images
    
    def file_path(file_dir,item):
        for file in os.listdir(file_dir):
            if str(item) in str(file):
                return str(os.path.join(file_dir,file))

default_content_dir = os.path.join(os.path.dirname(__file__), 'data', 'content')
default_style_dir = os.path.join(os.path.dirname(__file__), 'data', 'style')
