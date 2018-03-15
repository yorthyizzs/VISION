from PIL import Image
import numpy as np
import glob
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox

## These projection and image added plotting codes are taken from
## https://stackoverflow.com/questions/48180327/matplotlib-3d-scatter-plot-with-images-as-annotations
def proj(X, ax1, ax2):
    """ From a 3D point in axes ax1,
        calculate position in 2D in ax2 """
    x,y,z = X
    x2, y2, _ = proj3d.proj_transform(x,y,z, ax1.get_proj())
    return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))

def image(ax,arr,xy,):
    """ Place an image (arr) as annotation at position xy """
    im = offsetbox.OffsetImage(arr, zoom=0.08)
    im.image.axes = ax
    ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                        xycoords='data', boxcoords="offset points",
                        pad=0.3, arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)


####################################################################################################
def plot3D(values, images):
    zs = values[2]
    xs = values[1]
    ys = values[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter(xs, ys, zs, marker="o")
    # Create a dummy axes to place annotations to
    ax2 = fig.add_subplot(111, frame_on=False)
    ax2.axis("off")

    for x,y,z, name in zip(xs, ys, zs, images):
        x1, y1 = proj((x,y,z), ax, ax2)
        image(ax2, plt.imread(name), [x1, y1])
        ax.text(x,y,z,name.split('/Aligned_Fighter')[1].split('.bmp')[0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot2D(values, images):
    xs = values[1]
    ys = values[0]
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    for x,y,name in zip(xs,ys,images):
        image(ax, plt.imread(name), [x, y])
        ax.text(x, y, name.split('/Aligned_Fighter')[1].split('.bmp')[0])
    plt.show()

def plot1D(values, images):
    xs = values[0]
    ys = np.zeros(xs.shape)
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    for x, y, name in zip(xs, ys, images):
        image(ax, plt.imread(name), [x, y])
        ax.text(x, y, name.split('/Aligned_Fighter')[1].split('.bmp')[0])
    plt.show()

def reduce_dim(datapath):
    imList = []
    nameList = []

    for filename in glob.iglob(datapath+'/*.bmp'):
        im = Image.open(filename)
        nameList.append(filename)
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
    e_vecs2 = e_vecs[sorted_index[:2]]
    e_vecs1 = e_vecs[sorted_index[:1]]

    #represent data with 3 point
    VD = e_vecs3.dot(D.transpose())
    mapped = VD.dot(M)
    plot3D(mapped, nameList)


    # represent data with 2 point
    VD = e_vecs2.dot(D.transpose())
    mapped = VD.dot(M)
    #plt.scatter(mapped[0], mapped[1])
    plot2D(mapped, nameList)


    # represent data with 1 point
    VD = e_vecs1.dot(D.transpose())
    mapped = VD.dot(M)
    plot1D(mapped, nameList)

reduce_dim('Part1-dataset')


