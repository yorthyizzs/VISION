import json
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

data = json.load(open('train_set.json'))
imnames = []
labels = []

dirims = []
for imname in glob.glob('images/*.jpg'):
    im_id = imname.split('/')[1]
    dirims.append(im_id)

count = 0
for val, label in zip(data['images'],data['annotations']):
    imnames.append(val['id']+'.jpg')
    labels.append(label['category_id'])
   # if imnames[-1] not in dirims:
    #    print(imnames[-1])
     #   count +=1
#print(count)


"""

X_train, X_test, y_train, y_test = train_test_split(imnames, labels, test_size=0.33, random_state=42)
os.makedirs('train')
os.makedirs('val')
for im, lbl in zip(X_train, y_train):
    if not os.path.exists('train/'+str(lbl)):
        os.makedirs('train/'+str(lbl))
    try:
        shutil.copy2('images/'+im, 'train/'+str(lbl))
    except FileNotFoundError:
        print(im)
        pass


for im, lbl in zip(X_test, y_test):
    if not os.path.exists('val/'+str(lbl)):
        os.makedirs('val/'+str(lbl))
    shutil.copy2('images/'+im, 'val/'+str(lbl))

"""

