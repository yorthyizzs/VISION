import json
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

data = json.load(open('train_set.json'))
imnames = []
labels = []

for val, label in zip(data['images'], data['annotations']):
    imnames.append(val[u'id']+'.jpg')
    labels.append(label['category_id'])


os.makedirs('data')
X_train, X_test, y_train, y_test = train_test_split(imnames, labels, test_size=0.33, random_state=42)
os.makedirs('data/train')
os.makedirs('data/val')
for im, lbl in zip(X_train, y_train):
    if not os.path.exists('data/train/'+str(lbl)):
        os.makedirs('data/train/'+str(lbl))
    try:
        shutil.copy2('train_val/'+im, 'data/train/'+str(lbl))
    except FileNotFoundError:
        print(im)
        pass


for im, lbl in zip(X_test, y_test):
    if not os.path.exists('data/val/'+str(lbl)):
        os.makedirs('data/val/'+str(lbl))
    shutil.copy2('train_val/'+im, 'data/val/'+str(lbl))



