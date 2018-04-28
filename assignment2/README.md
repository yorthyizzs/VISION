# Scene Classification with Convolutional Neural Networks
* Python version : 3.5
* Torch version : 0.3.1
* Torchvision version : 0.2.1

There is 3 main script for each model.
Resnet18 and Vgg16 Parameters are like following : 

* argv[1]= epoch 
* argv[2] = batch size
--optionals:
* argv[3] = freezed parameter number(this should be 2*layer which will freeze)
(default = train only changed layer)
* argv[4] = learning rate
(default = 0.001)

--For alexnet there is no freezed parameter numbers as it's training from the stracth. So learning rate is argv[3] and the rest is same.

Before running and model script you should arrange the files to make usable for model scripts by running movefile script like following:

```sh
$ python3 movefile.py
```

After that feel free to use any models. Examples are like following
```sh
$ python3 vgg16.py 100 16 26 0.001
$ python3 alexnet.py 100 16 0.001
$ python3 resnet18.py 100 16 26 0.001
```


# References Of Codes That Helped
http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840

