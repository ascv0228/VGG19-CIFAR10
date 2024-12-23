import torch

BATCH_SIZE = 100
nepochs = 120
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')