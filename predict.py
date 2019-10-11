import torch
import numpy as np
from torch import nn
from torchvision import datasets, models, transforms
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('image_dir', help = 'Provide path to image', type = str)
parser.add_argument('load_dir', help = 'Provide path to checkpoint', type = str)
parser.add_argument('--top_k', help = 'To get top K most likely classes, enter a number (K). ', type = int)
parser.add_argument('--category_names', help = 'Provide JSON file name for mapping of categories to real names', type = str)
parser.add_argument('--gpu', help = "To choose to train the model on GPU, type cuda", type = str)

args = parser.parse_args()

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

def load_model(file_path):
    checkpoint = torch.load(file_path)
    
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        model = models.densenet121(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    p_image = Image.open(image) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    final_image = transform(p_image)
    
    return final_image


def predict(image_path, model, top_k, device):
    final_image = process_image(image_path)
    if device == 'cuda':
        final_image = final_image.type(torch.cuda.FloatTensor)
    else:
        final_image = final_image

    final_image.unsqueeze_(0)
    
    model.to(device)
    final_image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        ps = torch.exp(model.forward(final_image))
        top_probs, top_indices = ps.topk(top_k)
        
    top_probs = top_probs.cpu()
    top_indices = top_indices.cpu()
    
    top_probs = top_probs[0].numpy()
    
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_indices = top_indices[0].numpy()

    top_labels = []
    for idx in np_top_indices:
        top_labels.append(int(idx_to_class[idx]))
        
    return top_probs, top_labels

    
image_path = args.image_dir

model = load_model(args.load_dir)

if args.top_k:
    top_k = args.top_k
else:
    top_k = 1
    
if args.gpu == 'cuda':
    device = 'cuda'
else:
    device = 'cpu'

top_probs, top_labels = predict(image_path, model, top_k, device)

top_classes = [cat_to_name[str(lab)] for lab in top_labels]

for k in range(top_k):
     print("Number: {}/{}.. ".format(k+1, top_k),
            "Class name: {}.. ".format(top_classes[k]),
            "Probability: {:.2f} % ".format(top_probs[k]*100)
            )