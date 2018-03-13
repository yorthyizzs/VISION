import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class ImageRetrieval:
    def __init__(self, dataPath):
        self.datapath = dataPath
        self.query_File = 'QUERY_IMAGES'
        self.rgb_images = []
        self.grey_images = []
        self.hsv_images = []
        self.classes = []
        self.img_names = []
        self.filterBank = []
        self.gaborParams = []
        self.class_index_dict = {}
        self.rgb_histogram = None # Feature 1
        self.gabor_histogram = None # Feature 2
        self.hsv_histogram = None # Feature 3

    def _saveData(self):
        i = 0
        for dir in glob.iglob(self.datapath + '/*'):
            if self.query_File not in dir:
                for filename in glob.iglob(dir + '/*.jpg'):
                    im = self._imread(filename)
                    self.rgb_images.append(self._imConvert(im))
                    self.hsv_images.append(self._imConvert(self._rgbToHsv(im)))
                    self.grey_images.append(self._greyScale(im))
                    self.img_names.append(filename)
                    self.classes.append(i)
                    # following data is just for evaluation it's holds (class name - class index) key-value
                    self.class_index_dict[filename.split('/')[1]] = i
                i += 1
        self.classes = np.array(self.classes)

    def train(self):
        print("train has begun")
        self._saveData()
        print("data saved")
        self._histogramOfRgbs()
        print("histogramOfRgbs Saved")
        self._histogramOfHsvs()
        print("histogramOfHsvs Saved")
        self._genFilterBank()
        print("filter bank generated")
        self._histogramOfGabors()

    def query(self, query_image, type, target=-1, k=10):
        q_im = self._imread(query_image)
        distance = None

        if type == 'color_hist':
            q_im_rgb = self._imConvert(q_im)
            q_im_hist = np.concatenate((self._histogram(q_im_rgb[0]),
                                        self._histogram(q_im_rgb[1]),
                                        self._histogram(q_im_rgb[2])))
            distance = self._euclidianDistance(q_im_hist, self.rgb_histogram)
        elif type == 'hsv':
            q_im_hsv = self._imConvert(self._rgbToHsv(q_im))

            q_im_hsv_hist = np.concatenate((self._histogram(q_im_hsv[0]),
                                            self._histogram(q_im_hsv[1]),
                                            self._histogram(q_im_hsv[2])))
            distance = self._euclidianDistance(q_im_hsv_hist, self.hsv_histogram)
        elif type == 'gabor':
            q_grey = self._greyScale(q_im)
            q_filt1 = self._histogram(cv2.filter2D(q_grey, cv2.CV_8UC3, self.filterBank[0]))
            q_filt2 = self._histogram(cv2.filter2D(q_grey, cv2.CV_8UC3, self.filterBank[2]))
            q_filt3 = self._histogram(cv2.filter2D(q_grey, cv2.CV_8UC3, self.filterBank[4]))
            q_filt4 = self._histogram(cv2.filter2D(q_grey, cv2.CV_8UC3, self.filterBank[6]))
            q_gabor = np.concatenate((q_filt1, q_filt2, q_filt3, q_filt4))
            distance = self._euclidianDistance(q_gabor, self.gabor_histogram)

        rank = np.argsort(distance)[:k]

        if target != -1:
            ap = self.calculateAveragePrecision(rank, target, k)
            return ap
        else :
            for index in rank:
                print(self.img_names[index])

    def calculateMap(self, type, k):
        averagePrecisions = []
        for filename in glob.iglob(self.datapath+'/'+self.query_File+'/*.jpg'):
            q_class = self.class_index_dict[filename.split(self.datapath+'/'+self.query_File+'/')[1].split('_')[0]]
            ap = self.query(filename, type, target=q_class, k=k)
            averagePrecisions.append(ap)
        return np.mean(averagePrecisions)

    def _imread(self, im):
        return plt.imread(im)

    def _imConvert(self, im):
        im = np.asarray(im)
        return im.reshape(3, im.shape[0], im.shape[1])

    def _imshow(self, im):
        plt.imshow(im)
        plt.waitforbuttonpress()

    def _greyScale(self, im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    def _rgbToHsv(self, im):
        return cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    def _histogram(self, im):
        h = np.zeros(256)
        for row in range(im.shape[0]):
            for col in range(im.shape[1]):
                h[im[row, col]] += 1
        return h

    def _euclidianDistance(self, im, images):
        return np.sqrt(np.sum((images - im) ** 2, axis=1))

    def _histogramOfRgbs(self):
        r_hist = [self._histogram(im[0]) for im in self.rgb_images]
        g_hist = [self._histogram(im[1]) for im in self.rgb_images]
        b_hist = [self._histogram(im[2]) for im in self.rgb_images]

        self.rgb_histogram = np.concatenate((r_hist, g_hist, b_hist), axis=1)

    def _histogramOfHsvs(self):
        h_hist = [self._histogram(im[0]) for im in self.hsv_images]
        s_hist = [self._histogram(im[1]) for im in self.hsv_images]
        v_hist = [self._histogram(im[2]) for im in self.hsv_images]

        self.hsv_histogram = np.concatenate((h_hist, s_hist, v_hist), axis=1)

    def _genFilterBank(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            self.filterBank.append(kern)

    def _histogramOfGabors(self):
        vertical = []
        right = []
        horizontal = []
        left = []

        for im in self.grey_images:
            vertical.append(self._histogram(cv2.filter2D(im, cv2.CV_8UC3, self.filterBank[0])))
            right.append(self._histogram(cv2.filter2D(im, cv2.CV_8UC3, self.filterBank[2])))
            horizontal.append(self._histogram(cv2.filter2D(im, cv2.CV_8UC3, self.filterBank[4])))
            left.append(self._histogram(cv2.filter2D(im, cv2.CV_8UC3, self.filterBank[6])))

        self.gabor_histogram = np.concatenate((vertical, right, horizontal, left), axis=1)

    def calculateAveragePrecision(self, rank, target_class, k):
        retrieved = rank[:k]
        precision = []
        trues = 0
        i = 0
        for index in retrieved:
            i += 1
            if self.classes[index] == target_class:
                trues += 1
                precision.append(trues / i)
        return np.mean(np.array(precision))
