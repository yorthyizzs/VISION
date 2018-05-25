import objects
import glob
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import svm

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
            if usebase
                with open('train.pickle', 'rb') as handle:
                    f1, f2, f3, f4, f5, f6, lbls = pickle.load(handle)
                traindata, trainlabels = self.__mergeFeatures(types,f1, f2, f3, f4, f5, f6), lbls

                with open('test.pickle', 'rb') as handle:
                    f1, f2, f3, f4, f5, f6, lbls = pickle.load(handle)
                testdata, testlabels =self.__mergeFeatures(types,f1, f2, f3, f4, f5, f6), lbls
            else:
                with open('train_b.pickle', 'rb') as handle:
                    f1, f2, f3, f4, f5, f6, lbls = pickle.load(handle)
                traindata, trainlabels = self.__mergeFeatures(types, f1, f2, f3, f4, f5, f6), lbls

                with open('test_b.pickle', 'rb') as handle:
                    f1, f2, f3, f4, f5, f6, lbls = pickle.load(handle)
                testdata, testlabels = self.__mergeFeatures(types, f1, f2, f3, f4, f5, f6), lbls

        else:
            self.__fit(usebase)
            traindata, trainlabels = self.__getFeatures(types)
            testdata, testlabels = self.__getFeatures(types, train=False)

        if model == 'knn':
            neigh = KNeighborsClassifier(n_neighbors=37)
            neigh.fit(traindata, trainlabels)
            results = neigh.predict(testdata)
            print('Accuracy results of KNN {}'.format(self.__accuracy(results, testlabels)))

        elif model == 'svm':
            clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
            clf.fit(traindata, trainlabels)
            results = clf.predict(testdata)
            print('Accuracy results of SVM {}'.format(self.__accuracy(results, testlabels)))

        elif model == 'perceptron':
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(traindata[0]), 100, 7), random_state=1)
            clf.fit(traindata, trainlabels)
            results = clf.predict(testdata)
            print('Accuracy results of Perceptron {}'.format(self.__accuracy(results, testlabels)))

    def saveAsPickle(self, usebase = True):
        types = ['mag', 'ang', 'vgg', 'resnet', 'mag_pca', 'ang_pca']
        if usebase:

            self.__fit(usebase=True)

            f1, f2, f3, f4, f5, f6, lbls = self.__getFeatures(types, picklesave=True)
            with open('train.pickle', 'wb') as handle:
                pickle.dump((f1, f2, f3, f4, f5, f6, lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)

            f1, f2, f3, f4, f5, f6, lbls = self.__getFeatures(types, train=False, picklesave=True)
            with open('test.pickle', 'wb') as handle:
                pickle.dump((f1, f2, f3, f4, f5, f6, lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            self.__fit(usebase=False)

            f1, f2, f3, f4, f5, f6, lbls = self.__getFeatures(types, picklesave=True)
            with open('train_b.pickle', 'wb') as handle:
                pickle.dump((f1, f2, f3, f4, f5, f6, lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)

                f1, f2, f3, f4, f5, f6, lbls = self.__getFeatures(types, train=False, picklesave=True)
            with open('test_b.pickle', 'wb') as handle:
                pickle.dump((f1, f2, f3, f4, f5, f6, lbls), handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        f1, f2, f3, f4, f5, f6, lbls = [], [], [], [], [], [], []
        for emotion in emotions:
            t1, t2, t3, t4, t5, t6, tlbls = emotion.getData()
            f1.extend(t1)
            f2.extend(t2)
            f3.extend(t3)
            f4.extend(t4)
            f5.extend(t5)
            f6.extend(t6)
            lbls.extend(tlbls)
        if not picklesave:
            return self.__mergeFeatures(types,f1,f2,f3,f4, f5, f6), lbls
        else:
            return f1,f2,f3,f4,f5,f6,lbls

    def __mergeFeatures(self, types, f1, f2, f3, f4, f5, f6):
        feats = []
        for i in range(len(f1)):
            feature = []
            if 'mag' in types: feature.extend(f1[i])
            if 'ang' in types: feature.extend(f2[i])
            if 'vgg' in types: feature.extend(f3[i])
            if 'resnet' in types: feature.extend(f4[i])
            if 'mag_pca' in types: feature.extend(f5[i])
            if 'ang_pca' in types: feature.extend(f6[i])
            feature = np.array(feature)
            feats.append((feature-np.min(feature))/(np.max(feature)-np.min(feature)))
        return feats


if __name__ == '__main__':
    #types = [ ['mag_pca', 'ang_pca'], ['vgg', 'resnet'], ['mag_pca', 'resnet'], ['ang_pca', 'resnet'], ['mag_pca', 'vgg'], ['ang_pca', 'vgg']]
    #types = ['vgg', 'resnet', 'mag_pca', 'ang_pca']
    #types = [['ang'], ['mag'], ['mag_pca'], ['ang_pca']]
    #types = [['vgg'], ['resnet'], ['mag_pca'], ['ang_pca'], ['ang'], ['mag']]
    types = [['vgg'], ['resnet'], ['mag_pca'], ['ang_pca']]
    #types=[['vgg', 'resnet', 'mag_pca', 'ang_pca']]
    m = Model('DATA')
    #m.saveAsPickle(usebase=True)
    for typeset in types:
        print("Results for {}".format(typeset))
        m.train(types=typeset, model='knn', fromPickle=True)
        m.train(types=typeset, model='svm', fromPickle=True)
        m.train(types=typeset, model='perceptron', fromPickle=True)




