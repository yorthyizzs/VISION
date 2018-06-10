import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix


with open('train_gabor_lbl.pickle', 'rb') as handle:
    (trainfeats, trainlabel) = pickle.load(handle)
with open('val_gabor_lbl.pickle', 'rb') as handle:
    (valfeats, vallabel) = pickle.load(handle)


def accuracy(labels, testLabels):
    correct = 0
    for lbl, res in zip(labels, testLabels):
        correct += 1 if lbl == res else 0
    return float(correct / len(labels))

def predict(model, traindata, trainlabels, testdata, testlabels):
    results = None
    if model == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(traindata, trainlabels)
        results = neigh.predict(testdata)
        print('Accuracy results of KNN {}'.format(accuracy(results, testlabels)))

    elif model == 'svm':
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
        clf.fit(traindata, trainlabels)
        results = clf.predict(testdata)
        print('Accuracy results of SVM {}'.format(accuracy(results, testlabels)))

    tn, fp, fn, tp = confusion_matrix(testlabels, results).ravel()
    print(tn, fp, fn, tp)


predict('knn', trainfeats, trainlabel, valfeats, vallabel)
predict('svm', trainfeats, trainlabel, valfeats, vallabel)
