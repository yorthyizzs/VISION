import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from genericTrainer import train_model
import pickle

if __name__ == '__main__':
    import sys

    model = 'vgg16'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = None

    if model == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        count = 0
        for name, param in model_ft.named_parameters():
            param.requires_grad = False
            count += 1
            if count == 150:
                break
    elif model == 'resnet152':
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        count = 0
        for name, param in model_ft.named_parameters():
            param.requires_grad = False
            count += 1
            if count == 456:
                break
    elif model == 'vgg16':
        model_ft = models.vgg16(pretrained=True)
        num_ftrs = model_ft.classifier._modules['6'].in_features
        model_ft.classifier._modules['6'] = nn.Linear(num_ftrs, 2)
        count = 0
        for name, param in model_ft.named_parameters():
            param.requires_grad = False
            count += 1
            if count == 26:
                break
    elif model == 'alexnet':
        model_ft = models.alexnet(num_classes=2)

    elif model == 'densenet':
        model_ft = models.densenet121(pretrained=True)
        count = 0
        for name, param in model_ft.named_parameters():
            param.requires_grad = False
            count += 1
            if count == 21:
                break


    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)
    print(acc)

    with open('{}.pickle'.format(model), 'wb') as handle:
        pickle.dump(model_ft, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(model_ft.state_dict(), '{}_model.pt'.format(model))
