import glob
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from PIL import Image
import torch.nn as nn
import json

def readFileNamesAndLabels():
    trainim = []
    valim = []
    trainlabel = []
    vallabel = []
    for fname in glob.glob('data/*'):
        for dirname in glob.glob(fname + '/*'):
            cname = dirname.split(fname + '/')[1]
            pname = fname.split('/')[1]
            if pname == 'train':
                for imname in glob.glob(dirname + '/*.jpg'):
                    trainim.append(imname)
                    trainlabel.append(cname)
            else:
                for imname in glob.glob(dirname + '/*.jpg'):
                    valim.append(imname)
                    vallabel.append(cname)
    return trainim, valim, trainlabel, vallabel

def extractFeatures(images, mname='resnet18'):
    scaler = transforms.Resize((224, 224))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensorer = transforms.ToTensor()
    model = None
    features = []
    if mname == 'resnet18':
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for image in images:
        img = Image.open(image).convert('RGB')
        t_img = Variable(normalizer(to_tensorer(scaler(img))).unsqueeze(0))
        pred = model(t_img)
        features.append(pred.data.numpy().flatten())
    return features

def saveTrainData():
    trainim, valim, trainlabel, vallabel = readFileNamesAndLabels()
    trainfeats = extractFeatures(trainim)
    valfeats = extractFeatures(valim)
    print(len(vallabel), len(valim),  len(valfeats))
    print(len(trainlabel), len(trainim), len(trainfeats))

    with open('train_feat_lbl.pickle', 'wb') as handle:
        pickle.dump((trainfeats, trainlabel), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('val_feat_lbl.pickle', 'wb') as handle:
        pickle.dump((valfeats, vallabel), handle, protocol=pickle.HIGHEST_PROTOCOL)

def saveTestData():
    data = json.load(open('test_set.json'))
    imnames = []
    ids = []
    for im in data['images']:
        imnames.append(im['file_name'])
        ids.append(im['id'])
    features = extractFeatures(imnames)
    with open('test_feats.pickle', 'wb') as handle:
        pickle.dump((features,ids ), handle, protocol=pickle.HIGHEST_PROTOCOL)


saveTestData()
