import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageChops
from sklearn.decomposition import PCA

class Emotion:
    def __init__(self, label):
        self.label = label
        self.datas = []

    def calcSampleNum(self):
        num = 0
        for data in self.datas:
            num += len(data.sequence)-1
        return num

    def addData(self, data):
        self.datas.append(data)

    def train(self, usebase=True):
        for data in self.datas:
            data.train(usebase=usebase)

    def getData(self):
        f1, f2, f3, f4, f5, f6 = [], [], [], [], [], []
        for data in self.datas:
            f1.extend(data.mag)
            f2.extend(data.ang)
            f3.extend(data.vgg)
            f4.extend(data.resnet)
            f5.extend(data.mag_pca)
            f6.extend(data.ang_pca)
        lbls = [self.label for i in range(len(f1))]
        return f1, f2, f3, f4, f5, f6, lbls


class Data:
    def __init__(self, base, sequence):
        self.base = base
        # as sometime glob does not read in sorted order for some cases
        # to hold the sequence ordered sorted method is used here
        self.sequence = sorted(sequence)
        self.ang = []
        self.mag = []
        self.vgg = []
        self.resnet = []
        self.ang_pca = []
        self.mag_pca = []

    def train(self, usebase=True):
        self.__opticFlowFeatures(usebase=usebase)
        self.__modelFeature(usebase=usebase)
        self.__modelFeature(usebase=usebase, model='resnet')

    def __extractFacial(self, im):
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        img = cv2.imread(im)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return -1, -1, -1, -1
        for (x, y, w, h) in faces:
            return x, y, w, h

    def __opticFlowFeatures(self, usebase=True):
        base_im = cv2.cvtColor(cv2.imread(self.base), cv2.COLOR_BGR2GRAY)
        pca = PCA(n_components=10)
        for i in range(1, len(self.sequence)):
            x, y, w, h = self.__extractFacial(self.sequence[i-1])
            t_base = base_im
            previous_im = cv2.cvtColor(cv2.imread(self.sequence[i-1]), cv2.COLOR_BGR2GRAY)
            next_im = cv2.cvtColor(cv2.imread(self.sequence[i]), cv2.COLOR_BGR2GRAY)
            if x > 0:
                previous_im = previous_im[y:y + h, x:x + w]
                next_im = next_im[y:y + h, x:x + w]
                t_base = t_base[y:y + h, x:x + w]
            if usebase:
                previous_im -= t_base
                next_im -= t_base
            previous_im = cv2.resize(previous_im, (224, 224))
            next_im = cv2.resize(next_im, (224, 224))
            hsv = np.zeros_like(cv2.cvtColor(previous_im, cv2.COLOR_GRAY2BGR))
            hsv[:, :, 1] = 255
            flow = cv2.calcOpticalFlowFarneback(previous_im, next_im, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[:, :, 0] = ang * 180 / np.pi / 2
            hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            ##### hold the pc of the features ###
            pca.fit(hsv[:, :, 0])
            self.ang_pca.append(pca.singular_values_)
            pca.fit(hsv[:, :, 1])
            self.mag_pca.append(pca.singular_values_)
            #######################################

            self.ang.append(np.array(hsv[:, :, 0]).flatten())
            self.mag.append(np.array(hsv[:, :, 1]).flatten())
            #rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def __modelFeature(self, usebase, model='vgg'):
        scaler = transforms.Resize((224, 224))
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensorer = transforms.ToTensor()
        modal = None

        if model == 'vgg':
            modal = models.vgg16(pretrained=True)
            modal.classifier = nn.Sequential(*list(modal.classifier.children())[:-1])

        elif model == 'resnet':
            modal = models.resnet152(pretrained=True)
            modal = nn.Sequential(*list(modal.children())[:-1])

        modal.eval()
        for p in modal.parameters():
            p.requires_grad = False

        base = Image.open(self.base).convert('RGB')
        feature_arr = []
        for i in range(1, len(self.sequence)):
            img = Image.open(self.sequence[i]).convert('RGB')
            x, y, w, h = self.__extractFacial(self.sequence[i])
            img = img.crop((x, y, x + w, y + h)) if x > 0 else img
            t_base = base.crop((x, y, x + w, y + h)) if x > 0 else base

            if usebase:
                img = ImageChops.subtract(img, t_base)

            t_img = Variable(normalizer(to_tensorer(scaler(img))).unsqueeze(0))
            preds = modal(t_img)
            feature_arr.append(preds.data.numpy().flatten())

        if model == 'vgg':
            self.vgg = feature_arr
        elif model == 'resnet':
            self.resnet = feature_arr


