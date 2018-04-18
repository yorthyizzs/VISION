import os
import shutil
trainfile = open('dataset/train.txt', 'r')
testfile = open('dataset/test.txt', 'r')


for line in trainfile.readlines():
    line = line.strip()
    if not os.path.exists('train/'+line.split('/')[0]):
        os.makedirs('train/'+line.split('/')[0])
    shutil.copy2('dataset/Images/'+line, 'train/'+line)


for line in testfile.readlines():
    line = line.strip()
    if not os.path.exists('test/'+line.split('/')[0]):
        os.makedirs('test/'+line.split('/')[0])
    shutil.copy2('dataset/Images/'+line, 'test/'+line)


classes = ['airport_inside', 'bakery', 'bar', 'bedroom', 'bookstore',
           'children_room', 'classroom', 'clothingstore', 'computerroom',
           'concert_hall', 'dentaloffice', 'dining_room', 'grocerystore',
           'gym', 'hairsalon', 'inside_bus', 'kitchen', 'library',
           'prisoncell', 'subway']
