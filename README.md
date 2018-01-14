# Awesome-CNN-in-Keras
Implement awesome CNNs with Keras

## requirements
Keras2.1.0+, Tensorflow1.4+

## files
1.resnet_cifar.py
#### usage

nb_classes: the number of your dataset classes, for cifar-10, nb_classes should be 10

img_dim: the input shape of the model input

nb_blocks: the number of blocks in each stage

k: the widen fatcor, k=1 indicates that the model is original ResNet, when k>1 the model is a wide ResNet

weight_decay: weight decay for L2 regularization

droprate: the dropout between two convolutons of each block is added and the default drop rate is set to 0.0

ResNet model or WRN model

##### This is an example for ResNet-110:

from resnet_cifar import create_ResNet

resnet = create_ResNet(

                        nb_classes = 10,
                        
                        img_dim = (32, 32, 3),
                        
                        nb_blocks = [18, 18, 18],
                        
                        k = 1,
                        
                        droprate = 0.0
                        )
