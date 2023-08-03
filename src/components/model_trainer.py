import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import data_ingestion as DI
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import os

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.selected_features = ['0','5','10','19','28']
        self.model=models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.selected_features):
                features.append(x)
        return features

class modelTrain:
    def __init__(self,content_img_path,style_img_path,device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.content_img = DI.DataIngestion().load_image(content_img_path)
        self.style_img = DI.DataIngestion().load_image(style_img_path)
        self.generated = self.content_img.clone().requires_grad_(True)
        self.model = VGG().to(self.device).eval()

    def calc_content_loss(self,gen_feat,orig_feat):
        content_l=torch.mean((gen_feat-orig_feat)**2)
        return content_l
    
    def calc_style_loss(self,gen,style):

        batch_size,channel,height,width=gen.shape

        G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
        A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        
        style_l=torch.mean((G-A)**2)#/(4*channel*(height*width)**2)
        return style_l
    
    def calculate_loss(self,gen_features, orig_features, style_features,alpha,beta):
        style_loss=content_loss=0
        for gen,cont,style in zip(gen_features,orig_features,style_features):
            #extracting the dimensions from the generated image
            content_loss+=self.calc_content_loss(gen,cont)
            style_loss+=self.calc_style_loss(gen,style)
        
        #calculating the total loss of e th epoch
        total_loss=alpha*content_loss + beta*style_loss 
        return total_loss
    
    def train(self,epochs=100,lr=0.009,alpha=10,beta=700):
        optimizer = optim.Adam([self.generated],lr=lr)
        for e in range(epochs):
            gen_features = self.model(self.generated)
            cont_features = self.model(self.content_img)
            stl_features = self.model(self.style_img)

            total_loss = self.calculate_loss(gen_features,cont_features,stl_features,alpha=alpha,beta=beta)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if e%10 == 0 :
                print(f'Epoch [{e + 1}/{epochs}], Total Loss: {total_loss.item()}')

                generated_image = self.generated.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
                plt.imshow(generated_image)
                plt.axis('off')
                plt.title(f'Generated Image (Epoch {e + 1})')
                plt.show()

            

            


if __name__ == "__main__":
    content_dir = DI.default_content_dir
    style_dir = DI.default_style_dir
    
    content_img_1 = DI.DataIngestion.file_path(content_dir,'swans.jpg')
    style_img_1 = DI.DataIngestion.file_path(style_dir,'anime')

    print(content_img_1,style_img_1)

    test = modelTrain(content_img_1,style_img_1)
    test.train()