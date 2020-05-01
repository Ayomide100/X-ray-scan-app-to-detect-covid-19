import io

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

def get_model():
	checkpoint_path = 'checkpoint.pth'
	model = models.resnet18(pretrained = True)
	for param in model.parameters():
	    param.requires_grad = False
	model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 100)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.6)),
                          ('fc2', nn.Linear(100, 2)),
                          ('output', nn.LogSoftmax(dim=1)),
                          ]))
	
	model.load_state_dict(torch.load(
		checkpoint_path, map_location='cpu'), strict=False)
	model.eval()
	return model

def get_tensor(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ColorJitter(contrast=0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


	image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
	return my_transforms(image).unsqueeze(0)