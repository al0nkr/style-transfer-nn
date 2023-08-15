from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import components.data_ingestion as DI
import components.data_transformation as DT


def plot_img(img_path,trf = None,title = None,normalize = False):
    image = Image.open(img_path)
    if trf is not None:
        image = trf(image).unsqueeze(0)
        image = transforms.ToPILImage()(image.squeeze(0))
    
    if normalize:
        trf_normalize = DT.normalized_transform()
        image = trf_normalize(image)
    
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    
    image_array = np.array(image).astype(np.float32)# / 255.0
    image_array = image_array.transpose((1, 2, 0))
    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.show()

