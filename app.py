# import standard PyTorch modules
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models



train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ColorJitter(contrast=0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



data_dir = 'C:/Users/Test/Desktop/covid-19/images'

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


#iterating into the data
images, labels = next(iter(trainloader))

print(images.shape) #shape of all 4 images
print(images[1].shape) #shape of one imageS
print(labels[1].item()) #label number


np.random.seed(0)

# Defining the network  
model = models.resnet18(pretrained = True)
print(model)

# Only train the classifier parameters, feature parameters are frozen
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(512, 100)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.6)),
                        ('fc2', nn.Linear(100, 2)),
                        ('output', nn.LogSoftmax(dim=1)),
                        ]))


criterion = nn.NLLLoss()  

optimizer = optim.Adam(model.fc.parameters(), lr = 0.001) 
epochs = 2
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for image, label in trainloader:
        steps += 1
        optimizer.zero_grad()
        
        logps = model.forward(image)
        loss = criterion(logps, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for image, label in testloader:
                    #inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(image)
                    batch_loss =criterion(logps, label)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 512,
              'output_size':2,
              'epochs': epochs,
              'batch_size': 32,
              'model': model,
              'classifier': model.fc,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
   
torch.save(checkpoint, 'checkpoint.pth')