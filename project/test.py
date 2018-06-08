import json
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from PIL import Image
import torch
from sklearn.neural_network import MLPClassifier


def model():
    data = json.load(open('test_set.json'))
    mname = 'resnet50'

    imnames = []
    for im in data['images']:
        imnames.append(im['file_name'])

    scaler = transforms.Resize((224, 224))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensorer = transforms.ToTensor()

    with open('models/{}.pickle'.format(mname), 'rb') as handle:
        model = pickle.load(handle)
    model.eval()
    fp = open('model.csv', 'w')
    fp.write('id,label\n')
    for im in imnames:
        img = Image.open(im).convert('RGB')
        t_img = Variable(normalizer(to_tensorer(scaler(img))).unsqueeze(0))
        pred = model(t_img)
        _, result = torch.max(pred, 1)
        fp.write('{},{}\n'.format(im, int(result)))


def perceptron():
    with open('train_feat_lbl.pickle', 'rb') as handle:
        (trainfeats, trainlabel) = pickle.load(handle)

    with open('test_feats.pickle', 'rb') as handle:
        (features, ids) = pickle.load(handle)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(trainfeats[0]), 100, 7), random_state=1)
    clf.fit(trainfeats, trainlabel)
    results = clf.predict(features)
    fp = open('file.csv', 'w')
    fp.write('id,label\n')
    for res, id in zip(results, ids):
        fp.write('train_val/{}.jpg,{}\n'.format(id, res))

model()
