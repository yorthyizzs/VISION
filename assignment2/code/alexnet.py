from torchvision import models, transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch
import time
import copy



data_transforms = {
        'train': transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'test': transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

image_datasets = {x: datasets.ImageFolder(x, data_transforms[x]) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


def train(batchsize, num_epochs, lr):
    epoch_results = {}

    epoch_results['train-loss'] = []
    epoch_results['test-loss'] = []
    epoch_results['train-acc'] = []
    epoch_results['train-acc5'] = []
    epoch_results['test-acc'] = []
    epoch_results['test-acc5'] = []

    # GET THE VGG16 Pretrained model
    model = models.alexnet(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    softmax = nn.Softmax()

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True) for x in
                   ['train', 'test']}
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    since = time.time()
    best_acc = 0.0

    # best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_top1 = 0
            running_corrects_top5 = 0

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #outputs = softmax(outputs)
                _, preds = torch.max(outputs.data, 1)
                _, five_pred = outputs.topk(max((1, 5)), 1, True, True)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                for i in range(len(five_pred)):
                    running_corrects_top5 += int(sum(labels[i] == five_pred[i]).data)
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects_top1 += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects_top1 / dataset_sizes[phase]
            epoch_acc5 = running_corrects_top5 / dataset_sizes[phase]

            #collect the statistics
            if phase == 'train':
                epoch_results['train-loss'].append(epoch_loss)
                epoch_results['train-acc'].append(epoch_acc)
                epoch_results['train-acc5'].append(epoch_acc5)
            else:
                epoch_results['test-loss'].append(epoch_loss)
                epoch_results['test-acc'].append(epoch_acc)
                epoch_results['test-acc5'].append(epoch_acc5)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return epoch_results

if __name__ == '__main__':
    import sys

    epoch = int(sys.argv[1])
    batch = int(sys.argv[2])
    lr = 0.001
    if len(sys.argv) == 4:
        lr = float(sys.argv[3])

    results = train(batch, epoch,  lr)
    print("RESULTS FOR : epoch = {}, batch = {}, lr = {}".format(epoch,batch,lr))
    for key, val in results.items():
        print(key)
        print(val)
        print('-'*30)
