import pickle
import numpy as np
import hypertools as hyp
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


with open('train_feat_lbl.pickle', 'rb') as handle:
    (trainfeats, trainlabel) = pickle.load(handle)
with open('val_feat_lbl.pickle', 'rb') as handle:
    (valfeats, vallabel) = pickle.load(handle)

sc = StandardScaler()
X_std = sc.fit_transform(np.array(trainfeats))
pca = decomposition.PCA(n_components=3)
X_std_pca = pca.fit_transform(X_std)
hyp.plot(np.array(X_std_pca),'.', group=np.array(trainlabel))
hyp.plot(np.array(X_std_pca)[np.array(trainlabel) =='0'],'.')
hyp.plot(np.array(X_std_pca)[np.array(trainlabel) =='1'],'.', color='cyan')
