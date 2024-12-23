import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time
import os

from hyperparameters import *
from network import VGG19
from utils import *

loss_func = nn.CrossEntropyLoss()

mean = [x/255 for x in [125.3, 23.0, 113.9]] 
std = [x/255 for x in [63.0, 62.1, 66.7]]
n_train_samples = 50000

# transform = transforms.Compose(
#     [transforms.RandomHorizontalFlip(),
#      transforms.RandomVerticalFlip(),
#      transforms.RandomRotation(30),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )
checkpoints_folder = "./q1_checkpoints"


train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []


if __name__ == '__main__':
    # dataset
    train_set = datasets.CIFAR10(root='../data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(30),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                             ]))
    train_dl = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)
    # train_set.train_data = train_set.train_data[0:n_train_samples]
    # train_set.train_labels = train_set.train_labels[0:n_train_samples]

    test_set = datasets.CIFAR10(root='../data',
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ]))

    test_dl = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         num_workers=0)     
    
    vgg19 = VGG19().to(device)
    # print(vgg19)


    os.makedirs(checkpoints_folder, exist_ok=True)
    ##########################
    ### summary
    ##########################
    from torchinfo import summary

    summary(vgg19, input_size=(BATCH_SIZE, 3, 32, 32))



    optimizer = torch.optim.SGD(vgg19.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
    learn_history = []

    print('training VGG19……')

    for epoch in range(nepochs):
        since = time.time()
        train_epoch(vgg19, loss_func, optimizer, train_dl)

        
        tr_loss, tr_acc = eval(vgg19, loss_func, train_dl)
        te_loss, te_acc = eval(vgg19, loss_func, test_dl)
        learn_history.append((tr_loss, tr_acc, te_loss, te_acc))
        now = time.time()
        print('[%3d/%d, %.0f seconds]|\t tr_err: %.1e, tr_acc: %.2f\t |\t te_err: %.1e, te_acc: %.2f'%(
            epoch+1, nepochs, now-since, tr_loss, tr_acc, te_loss, te_acc))
        

    
        train_accuracies.append(tr_acc.item())
        train_losses.append(tr_loss)
        valid_accuracies.append(te_acc.item())
        valid_losses.append(te_loss)
    
        checkpoint_path = f'{checkpoints_folder}/vgg19_model_epoch_{epoch+1}.pth'
        torch.save(vgg19.state_dict(), checkpoint_path)

    
    print ("train_losses", ":" , train_losses)
    print ("valid_losses", ":" , valid_losses)
    print ("train_accuracies", ":" , train_accuracies)
    print ("valid_accuracies", ":" , valid_accuracies)

    plot_loss_acc(train_losses, valid_losses, train_accuracies, valid_accuracies)

    with torch.set_grad_enabled(False): # save memory during inference
        print('Test loss, accuracy: ', eval(vgg19, loss_func, test_dl))


# train_losses : [1.2242976552248002, 0.9036710883378982, 0.8619430331587792, 0.5905699228048324, 0.5343412283658981, 0.5203082274794578, 0.4489125165641308, 0.41604129755496977, 0.3803886116147041, 0.381542567551136, 0.313222259759903, 0.2924929533004761, 0.3875343539118767, 0.302925112888217, 0.25101562729477883, 0.2582785425633192, 0.20162556682527066, 0.2018875723183155, 0.2139654915481806, 0.18366847448050977, 0.17891439171135426, 0.1543256862461567, 0.17273983143270016, 0.18574705754220486, 0.1282605903595686, 0.12688749239593744, 0.14689900997281075, 0.12406071409210563, 0.09391374832764268, 0.10256629109568893, 0.10604415239207446, 0.09758997492678463, 0.0814341834783554, 0.07640165005251766, 0.07817707782238723, 0.07930009309202432, 0.06570961254090071, 0.07476613979693503, 0.06564169492945075, 0.05698908684495836, 0.05502305917721242, 0.05313176077697426, 0.05438422354310751, 0.05645725235482678, 0.04698818793147802, 0.04548426590533927, 0.048464168105972934, 0.040796122456435116, 0.04279760270705447, 0.03548772874288261]
# valid_losses : [1.194636394381523, 0.8910134333372116, 0.8986099034547805, 0.6157838302850723, 0.5784550151228904, 0.5542018330097198, 0.49507640570402145, 0.4861901861429214, 0.4744896155595779, 0.4772449438273907, 0.4194050766527653, 0.4222357799112797, 0.49736855879426, 0.45195255279541013, 0.404792068451643, 0.40240658447146416, 0.35990423560142515, 0.36494103133678435, 0.39619306221604345, 0.3748731617629528, 0.3695525169372559, 0.35299605041742327, 0.37921773925423624, 0.38527357667684553, 0.3527160277962685, 0.35133802220225335, 0.3968618175387382, 0.3520380640029907, 0.34461772717535494, 0.34558354869484903, 0.35370397329330444, 0.370615673288703, 0.35570630967617034, 0.327137633562088, 0.36776732221245767, 0.3859576915204525, 0.33266287833452224, 0.37060384452342987, 0.3681285519897938, 0.3395345202088356, 0.3591024573147297, 0.3459884639829397, 0.3600065377354622, 0.38539709195494654, 0.3692009404301643, 0.36782297544181347, 0.3931909714639187, 0.37557597547769545, 0.3735257901251316, 0.3872015859931707]
# train_accuracies : [56.349979400634766, 68.4840087890625, 70.62602233886719, 80.04598236083984, 81.99392700195312, 82.09593200683594, 84.99799346923828, 86.07000732421875, 87.10599517822266, 87.2879638671875, 89.4700698852539, 90.16004180908203, 87.1839828491211, 89.81208038330078, 91.7460708618164, 91.31608581542969, 93.1580810546875, 93.11209106445312, 92.5740737915039, 93.71006774902344, 93.97411346435547, 94.70610046386719, 94.14411926269531, 93.5500717163086, 95.65011596679688, 95.79613494873047, 94.9380874633789, 95.77808380126953, 96.86605834960938, 96.54212951660156, 96.4801254272461, 96.64610290527344, 97.18006134033203, 97.44212341308594, 97.36006164550781, 97.25613403320312, 97.85609436035156, 97.50005340576172, 97.75408935546875, 98.07805633544922, 98.05204772949219, 98.1939697265625, 98.2100601196289, 98.09002685546875, 98.45002746582031, 98.3821029663086, 98.38199615478516, 98.63801574707031, 98.50997924804688, 98.83992004394531]
# valid_accuracies : [57.40000534057617, 69.52999114990234, 69.93000030517578, 79.7400131225586, 80.87001037597656, 81.11000061035156, 83.65998077392578, 83.97999572753906, 84.32998657226562, 84.5099868774414, 86.25000762939453, 86.489990234375, 84.12999725341797, 85.40000915527344, 87.31999969482422, 87.29998016357422, 88.25, 88.2599868774414, 87.15001678466797, 88.25001525878906, 88.45997619628906, 88.79999542236328, 88.49000549316406, 88.22000885009766, 89.3600082397461, 89.68001556396484, 88.79000854492188, 89.67997741699219, 89.989990234375, 90.13999938964844, 89.82999420166016, 89.43998718261719, 90.2699966430664, 90.40003967285156, 90.17000579833984, 90.06000518798828, 90.44999694824219, 90.26001739501953, 90.22996520996094, 90.8599853515625, 90.57000732421875, 90.7399673461914, 90.35000610351562, 90.18000030517578, 90.72001647949219, 90.55001068115234, 90.71998596191406, 90.78998565673828, 90.8499984741211, 91.12998962402344]