import objects
import glob
import pickle
from sklearn.neighbors import KNeighborsClassifier

class Model:
    def __init__(self, file):
        self.classes = {}
        self.file = file
        self.trainEmotions = []
        self.testEmotions = []
        self.__collectData()

    def __collectData(self):
        for file, dataList in zip([self.file + '/TRAIN', self.file + '/TEST'], [self.trainEmotions, self.testEmotions]):
            neutral = {}
            for img in glob.glob(file+'/neutral/*.png'):
                neutral[img.split(file+'/neutral/')[-1].split('_00000001.png')[0]] = img
            for dir in glob.glob(file + '/*'):
                label = dir.split(file + '/')[1]
                if label != 'neutral':
                    if label not in self.classes.keys():
                        self.classes[label] = len(self.classes.keys())
                    emotion = objects.Emotion(self.classes[label])
                    for subdir in glob.glob(dir + '/*'):
                        for subsubdir in glob.glob(subdir + '/*'):
                            name = subdir.split(dir + '/')[1] + '_' + subsubdir.split(subdir + '/')[1]
                            imgs = []
                            for img in glob.glob(subsubdir + '/*.png'):
                               imgs.append(img)
                            try:
                                emotion.addData(data=objects.Data(base=neutral[name], sequence=imgs))
                            except:
                                # this is done for the missing neutral image in train data (S506_006)
                                emotion.addData(data=objects.Data(base=neutral['S506_004'], sequence=imgs))
                    dataList.append(emotion)

    def train(self, types, usebase=True, model='knn', fromPickle=False):
        if fromPickle:
            with open('train.pickle', 'rb') as handle:
                f1, f2, f3, lbls = pickle.load(handle)
            traindata, trainlabels = self.__mergeFeatures(types,f1,f2,f3), lbls

            with open('test.pickle', 'rb') as handle:
                f1, f2, f3, lbls = pickle.load(handle)
            testdata, testlabels =self.__mergeFeatures(types,f1,f2,f3), lbls

        else:
            self.__fit(usebase)
            traindata, trainlabels = self.__getFeatures(types)
            testdata, testlabels = self.__getFeatures(types, train=False)

        if model == 'knn':
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(traindata, trainlabels)
            results = neigh.predict(testdata)
            print('Accuracy results of KNN {}'.format(self.__accuracy(results, testlabels)))

    def saveAsPickle(self):
        types = ['mag', 'ang', 'vgg']
        self.__fit(usebase=True)

        f1, f2, f3, lbls = self.__getFeatures(types, picklesave=True)
        with open('train.pickle', 'wb') as handle:
            pickle.dump((f1,f2,f3,lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)

        f1, f2, f3, lbls = self.__getFeatures(types, train=False, picklesave=True)
        with open('test.pickle', 'wb') as handle:
            pickle.dump((f1,f2,f3,lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.__fit(usebase=False)

        f1, f2, f3, lbls = self.__getFeatures(types, picklesave=True)
        with open('train_b.pickle', 'wb') as handle:
            pickle.dump((f1, f2, f3, lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)

        f1, f2, f3, lbls = self.__getFeatures(types, train=False, picklesave=True)
        with open('test_b.pickle', 'wb') as handle:
            pickle.dump((f1, f2, f3, lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)


    def __accuracy(self, labels, testLabels):
        correct = 0
        for lbl, res in zip(labels, testLabels):
            correct += 1 if lbl == res else 0
        return float(correct / len(labels))

    def __fit(self, usebase=True):
        for emotion in self.trainEmotions:
            emotion.train(usebase)
        for emotion in self.testEmotions:
            emotion.train(usebase)

    def __getFeatures(self, types, train=True, picklesave=False):
        emotions = self.trainEmotions if train else self.testEmotions
        f1, f2, f3, lbls = [], [], [], []
        for emotion in emotions:
            t1, t2, t3, tlbls = emotion.getData()
            f1.extend(t1)
            f2.extend(t2)
            f3.extend(t3)
            lbls.extend(tlbls)
        if not picklesave:
            return self.__mergeFeatures(types,f1,f2,f3), lbls
        else:
            return f1,f2,f3,lbls

    def __mergeFeatures(self, types, f1, f2, f3):
        feats = []
        for i in range(len(f1)):
            feature = []
            if 'mag' in types: feature.extend(f1[i])
            if 'ang' in types: feature.extend(f2[i])
            if 'vgg' in types: feature.extend(f3[i])
            feats.append(feature)
        return feats


if __name__ == '__main__':
    types = [ 'vgg']
    m = Model('DATA')
    #m.saveAsPickle()
    m.train(types=types, fromPickle=True)


