import glob
from PIL import Image
import numpy as np
import pickle


data_path = 'Part2-dataset'
query_File = 'QUERY_IMAGES'

def histogram(im):
    h1 = np.zeros(256)
    h2 = np.zeros(256)
    h3 = np.zeros(256)
    for row in range(im.shape[1]):
        for col in range(im.shape[2]):
            h1[im[0, row, col]] += 1
            h2[im[1, row, col]] += 1
            h3[im[2, row, col]] += 1
    h = np.append(np.append(h1, h2), h3)
    return h

def getHistogramMatrix(images):
    hist = [histogram(im) for im in images]
    return np.array(hist)

def calculateAveragePrecision(rank, classes, target_class, k):
    retrieved = rank[:k]
    precision = []
    trues = 0
    i = 0
    for index in retrieved:
        i += 1
        if classes[index] == target_class:
            trues += 1
            precision.append(trues / i)
    return np.mean(np.array(precision))


def euclidianDistance(im, images):
    return np.sqrt(np.sum((images-im)**2, axis=1))

def getData(datapath, queryFile):
    # will contain all images  (#of images x rgb channels of image )
    imgs = []
    # will contain the number of relevant image
    # of that class (ex: we have 30 images related with airplane)
    # by holding the class variable in the same index with the image
    classes = []
    # name of each image in imgs list to use in future
    names = []
    i = 0
    for dir in glob.iglob(datapath + '/*'):
        if query_File not in dir:
            for filename in glob.iglob(dir + '/*.jpg'):
                im = Image.open(filename)
                im = np.asarray(im)
                imgs.append(im.reshape(3, im.shape[0], im.shape[1]))
                names.append(filename)
                classes.append(i)
            i += 1

    return imgs, np.array(classes), names

"""
imgs, classes = getData(data_path, query_File)
matrix = getHistogramMatrix(imgs)

with open('matrix.pickle', 'wb') as handle:
    pickle.dump(matrix, handle)"""

with open('matrix.pickle', 'rb') as handle:
    matrix = pickle.load(handle)
imgs, classes, names = getData(data_path, query_File)

im = Image.open('Part2-dataset/QUERY_IMAGES/airplane_251_0175.jpg')
im = np.asarray(im)
Q_image = im.reshape(3, im.shape[0], im.shape[1])
Q_hist = histogram(Q_image)

x = euclidianDistance(Q_hist, matrix)
rank = np.argsort(x)
ap=calculateAveragePrecision(rank, classes, 9, 50)
print(ap)


"""
('Part2-dataset/cactus', 0)
('Part2-dataset/blimp', 1)
('Part2-dataset/googse', 2)
('Part2-dataset/dog', 3)
('Part2-dataset/iris', 4)
('Part2-dataset/bear', 5)
('Part2-dataset/goat', 6)
('Part2-dataset/bonsai', 7)
('Part2-dataset/ibis', 8)
('Part2-dataset/airplane', 9)
"""
