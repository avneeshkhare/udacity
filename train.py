import torch
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets, models, transforms
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help = 'Provide data directory', type = str)
parser.add_argument('--save_dir', help = 'Provide saving directory', type = str)
parser.add_argument('--arch', help = 'To choose architecture, type alexnet. Default is densenet121.', type = str)
parser.add_argument('--learning_rate', help = 'Learning rate', type = float)
parser.add_argument('--hidden_units', help = 'Number of hidden units', type = int)
parser.add_argument('--epochs', help = 'Number of training epochs', type = int)
parser.add_argument('--gpu', help = "To choose to train the model on GPU, type cuda", type = str)

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.gpu == 'cuda':
    device = 'cuda'
else:
    device = 'cpu'

train_transforms = transforms.Compose([transforms.RandomRotation (30),
                                       transforms.RandomResizedCrop (224),
                                       transforms.RandomHorizontalFlip (),
                                       transforms.ToTensor (),
                                       transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                       ])

valid_transforms = transforms.Compose([transforms.Resize (255),
                                       transforms.CenterCrop (224),
                                       transforms.ToTensor (),
                                       transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                       ])

test_transforms = transforms.Compose([transforms.Resize (255),
                                      transforms.CenterCrop (224),
                                      transforms.ToTensor (),
                                      transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])

train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)

 
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)


def load_model(arch, hidden_units):
    if arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            model.classifier = nn.Sequential(nn.Linear(9216, 4096),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(4096, hidden_units),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_units, 102),
                                             nn.LogSoftmax(dim=1))
        else:
            model.classifier = nn.Sequential(nn.Linear(9216, 4096),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(4096, 2048),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(2048, 102),
                                             nn.LogSoftmax(dim=1))
    else:
        arch = 'densenet121'
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(512, hidden_units),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_units, 102),
                                             nn.LogSoftmax(dim=1))
        else:
            model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(256, 102),
                                             nn.LogSoftmax(dim=1))    
        
        
    return model, arch

model, arch = load_model(args.arch, args.hidden_units)
criterion = nn.NLLLoss()

if args.learning_rate:
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.002)

model.to(device)

if args.epochs:
    epochs = args.epochs
else:
    epochs = 7
    
steps = 0
running_loss = 0
print_every = 50
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {(accuracy/len(validloader))*100:.2f} %")
            running_loss = 0
            model.train()

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {(accuracy/len(testloader))*100:.2f} %")
    
    
model.to('cpu')
model.class_to_idx = train_data.class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
              'arch': arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict()
             }

if args.save_dir:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')