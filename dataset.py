import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from models import *
from utils import tokenize, tokenizesst5
from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset

def dataset_init(dataset, device, batch_size=32, nt=32, ns=1, identical=True):
    if identical:
        if dataset == 'cifar100':
        
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Mean and std for CIFAR-100
            ])

            teacher_constituents = [models.resnet50(pretrained=False) for _ in range(nt)]
            for model in teacher_constituents:
                model.fc = nn.Linear(model.fc.in_features, 100)
                model.to(device)

            student_constituents = [models.resnet50(pretrained=False) for _ in range(ns)]
            for model in student_constituents:
                model.fc = nn.Linear(model.fc.in_features, 100)
                model.to(device)

            trainset = torchvision.datasets.CIFAR100(
                root="./dataset/cifar-100-python/", train=True, download=True, transform=transform
            )

            testset = torchvision.datasets.CIFAR100(
                root="./dataset/cifar-100-python/", train=False, download=True, transform=transform
            )
    
        elif dataset == 'MNIST':
            batch_size = 512
            transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Normalize((0.1307,), (0.3081,))  # Mean & Std Dev for MNIST
            ])

            teacher_constituents = [CNN_MNIST() for _ in range(nt)]
            for model in teacher_constituents:
                model.to(device)

            student_constituents = [CNN_MNIST() for _ in range(ns)]
            for model in student_constituents:
                model.to(device)

            trainset = torchvision.datasets.MNIST(root="./dataset/MNIST/", train=True, download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root="./dataset/MNIST/", train=False, download=True, transform=transform)

        elif dataset == 'SVHN':
            transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for SVHN
            ])

        
            teacher_constituents = [CNN_SVHN() for _ in range(nt)]
            for model in teacher_constituents:
                model.to(device)

            student_constituents = [CNN_SVHN() for _ in range(ns)]
            for model in student_constituents:
                model.to(device)


            # Load training data
            trainset = torchvision.datasets.SVHN(root="./dataset/SVHN/", split='train', download=True, transform=transform)

            # Load test data
            testset = torchvision.datasets.SVHN(root="./dataset/SVHN/", split='test', download=True, transform=transform)

        elif dataset == 'sst5':
            ds = load_dataset("SetFit/sst5")
            ds = ds.map(tokenizesst5, batched=True)
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            ds = ds.rename_column("label", "labels")

            trainset = ds['train']
            testset = ds['validation']
                
            teacher_constituents = [BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5) for _ in range(nt)]
            for model in teacher_constituents:
                model.to(device)
            student_constituents = [BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5) for _ in range(ns)]
            for model in student_constituents:
                model.to(device)
        else:
            raise Exception("Dataset Not Supported")

    return trainset, testset, teacher_constituents, student_constituents, batch_size
