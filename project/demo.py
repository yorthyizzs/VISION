import pickle
import json
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np

softmax = nn.Softmax()
mname = 'resnet50'
with open('models/{}.pickle'.format(mname), 'rb') as handle:
    model = pickle.load(handle)

model.eval()

data = json.load(open('test_set.json'))
imnames = []
for im in data['images']:
    imnames.append(im['file_name'])

scaler = transforms.Resize((224, 224))
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensorer = transforms.ToTensor()

labels = []
imgs = []
for i in range(4):
    im = random.choice(imnames)
    img = Image.open(im).convert('RGB')
    imgs.append(img)
    t_img = Variable(normalizer(to_tensorer(scaler(img))).unsqueeze(0))
    pred = model(t_img)
    _, result = torch.max(pred, 1)
    percentage,_ = torch.max(softmax(pred),1)
    label = 'Not Exist' if int(result) == 0 else 'Exist'
    labels.append(label)
    ax = plt.subplot("{}".format(221+i))
    ax.set_title(label+('%.4f' %percentage))
    ax.imshow(img)



plt.waitforbuttonpress()
