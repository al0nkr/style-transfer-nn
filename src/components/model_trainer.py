import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import src.components.data_ingestion as DI
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.selected_features = ['0','5','10','19','28']
        self.model=models.vgg19(pretrained=True).features[:29]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self,x):
        x = (x - self.mean) / self.std

        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if (str(layer_num) in self.selected_features):
                features.append(x)
        return features

class modelTrain:
    def __init__(self,content_img_path,style_img_path,device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.content_img = DI.DataIngestion().load_image(content_img_path).to(self.device)
        self.style_img = DI.DataIngestion().load_image(style_img_path).to(self.device)
        self.generated = self.content_img.clone().requires_grad_(True).to(self.device)
        self.model = VGG().to(self.device).eval()

    def calc_content_loss(self,gen_feat,orig_feat):
        content_l=torch.mean((gen_feat-orig_feat)**2)
        return content_l
    
    def calc_style_loss(self,gen,style):

        _,channel,height,width=gen.shape

        G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
        A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        
        style_l=torch.mean((G-A)**2)
        return style_l
    
    def calculate_loss(self, gen_features, orig_features, style_features, alpha, beta):
        style_loss = content_loss = 0
        num_features = len(gen_features)

        def calc_content_style_loss(idx):
            gen, cont, style = gen_features[idx], orig_features[idx], style_features[idx]
            content_loss = self.calc_content_loss(gen, cont)
            style_loss = self.calc_style_loss(gen, style)
            return content_loss, style_loss
        
        with ThreadPool() as pool:
            results = pool.map(calc_content_style_loss, range(num_features))

        for content_loss_i, style_loss_i in results:
            content_loss += content_loss_i
            style_loss += style_loss_i

        total_loss = alpha * content_loss + beta * style_loss
        return total_loss, content_loss, style_loss
    
    def train(self,epochs=100,lr=0.009,alpha=8,beta=70):
        optimizer = optim.Adam([self.generated],lr=lr)
        start_time = time.time()
        for e in range(epochs):
            gen_features = self.model(self.generated)
            cont_features = [cont_feat.detach() for cont_feat in self.model(self.content_img)]
            stl_features = [stl_feat.detach() for stl_feat in self.model(self.style_img)]
            total_loss, content_loss, style_loss = self.calculate_loss(gen_features, cont_features, stl_features, alpha=alpha, beta=beta)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()
            end_time = time.time()
            elapsed_time = end_time - start_time

            #if (e+1)%10 == 0 :
        print(f'Epoch [{e + 1}/{epochs}], Content Loss: {content_loss.item()} , Style Loss: {style_loss.item()}, Total Loss: {total_loss.item()} , Time for {e + 1} epochs : {elapsed_time:.02f}')
        self.gen_output_image = self.generated.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        #plt.imshow(self.gen_output_image)
        #plt.axis('off')
        #plt.title(f'Generated Image (Epoch {e + 1})')
        #plt.show()
        return self.gen_output_image


if __name__ == "__main__":
    content_dir = DI.default_content_dir
    style_dir = DI.default_style_dir
    
    content_img_1 = DI.DataIngestion.file_path(content_dir,'swans')
    style_img_1 = DI.DataIngestion.file_path(style_dir,'red')

    print(content_img_1,style_img_1)

    """num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Pass the function and its arguments to the Pool to run in parallel
        result = pool.starmap(modelTrain(content_img_1, style_img_1).train())"""

    test = modelTrain(content_img_1,style_img_1)
    test.train()