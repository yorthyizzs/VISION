import pickle
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import hypertools as hyp
from hypertools import tools
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from sklearn import svm

#pca = PCA(n_components=512)

with open('train_feat_lbl.pickle', 'rb') as handle:
    (trainfeats, trainlabel) = pickle.load(handle)
with open('val_feat_lbl.pickle', 'rb') as handle:
    (valfeats, vallabel) = pickle.load(handle)

#percentage = int(len(vallabel)  * 0.2)
#valfeats, vallabel = shuffle(valfeats, vallabel, random_state=0)
#tools.describe.describe(np.array(valfeats), reduce='PCA', max_dims=14)
#feats =np.array(valfeats[:percentage])
#lbls = np.array(vallabel[:percentage])
hyp.plot(np.array(valfeats),'*', group=np.array(vallabel))

