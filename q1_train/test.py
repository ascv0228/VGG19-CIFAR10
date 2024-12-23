import os
import torch
from torchvision import transforms

from PIL import Image
from hyperparameters import *
from network import VGG19
from utils import *

image_folder = "../../cvdl_hw2_data/Dataset_CvDl_Hw2/Q1_image/Q1_4"

image_list = os.listdir(image_folder)
print(image_list)

img_list = map(lambda x : Image.open(os.path.join(image_folder, x)).convert('RGB'), image_list)

mean = [x/255 for x in [125.3, 23.0, 113.9]] 
std = [x/255 for x in [63.0, 62.1, 66.7]]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
image_tensor_list = [ transform(i).unsqueeze(0).to(device) for i in img_list]

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
best_checkpoint = {9:[], 10:[]}
for epoch in range(15, 121):
    
    vgg19 = VGG19().to(device)
    checkpoint_path = f'./q1_checkpoints/vgg19_model_epoch_{epoch}.pth'
    vgg19.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vgg19.eval()
    print(epoch)
    
    with torch.no_grad(): 
        k = 0
        for i in range(len(image_tensor_list)):
            output = vgg19(image_tensor_list[i]) 
            predicted_class = torch.argmax(output, dim=1).item() 
            if (image_list[i] == cifar10_classes[predicted_class]+".png"):
                k+=1
        else:
            if k >= 9:
                best_checkpoint[k].append(checkpoint_path)

print(10, best_checkpoint[10])
print(9, best_checkpoint[9])

