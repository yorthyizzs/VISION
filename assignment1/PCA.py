from PIL import Image
import numpy as np
import glob
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

imList = []
for filename in glob.iglob('Part1-dataset/*.bmp'):
    im = Image.open(filename)
    imList.append(im.getdata())
# 65536x11 feature matrix
M = np.array(imList).transpose()
# 11x1 mean of each image
mean_vector = M.mean(axis=0)
# 65536x11 normalized data matrix
D = M - mean_vector
# 11x11 covariance matrix
cov = M.transpose().dot(M)
# eigenvectors(11x11) and eigenvalues(11x1) of covariance matrix
e_vals, e_vecs = LA.eig(cov)
# first thee eigenvalues's indexes
sorted_index = np.argsort(e_vals)
# those value's eigenvectors
e_vecs3 = e_vecs[sorted_index[:3]]

VD = e_vecs3.dot(D.transpose())
last = VD.dot(M)
zdata = last[2]
xdata = last[1]
ydata = last[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xdata, ydata, zdata)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
