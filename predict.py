from argparse import ArgumentParser
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
import keras.applications as apps
import pickle
import cv2
import os
import functools
                       
from keras.applications.vgg16 import preprocess_input

# preprocessing functions dictonary
input_preprocessing = {
    'resnet50V2': apps.resnet_v2.preprocess_input,
    'mobileNetV3Small': apps.mobilenet_v3.preprocess_input,
    'densenet121': apps.densenet.preprocess_input,
    'nasnetMobile': apps.nasnet.preprocess_input,
    'efficientNetB0': apps.efficientnet.preprocess_input
}


# available architectures
models_list = [
    'efficientNetB0',
    'nasnetMobile',
    'mobileNetV3Small',
    'densenet121',
    'resnet50V2'
]

parser = ArgumentParser()
parser.add_argument('model', help='which model to use', type=str, choices=models_list)
parser.add_argument('pathModel', help='path to load model (do not use the extension file)', type=str)
parser.add_argument('pathData', help='path to image (test)', type=str)

args = parser.parse_args()


modelPath = args.pathModel 	# /model/model'
imagePath = args.pathData 	# '/images/Test/16730/197886.jpg'


def getTopK(answer: np.array, class_list: list):
    '''Get answer'''
    if answer > 0.5: # threshold
        return [(class_list[1], answer)]
    else: 
        return [(class_list[0], answer)]


with open(modelPath + '.bin', 'rb') as class_file:
    modelName, classes = pickle.load(class_file)
if isinstance(classes, LabelBinarizer):
    classes = classes.classes_
elif isinstance(classes, OneHotEncoder):
    classes = classes.classes
else:
    raise TypeError('Classes object type is not supported ({}).'.format(type(classes).__name__))


# Top-1 metric
top1 = functools.partial(top_k_categorical_accuracy, k=1)
top1.__name__ = 'top1'
# Top-5 metric
top5 = functools.partial(top_k_categorical_accuracy, k=5)
top5.__name__ = 'top5'

#model
print('\nModel:' + args.model)

#image
print('\nTest image: ' + imagePath  + '\n')

#load model
print('Loading model: ' + modelPath  + '.h5\n')
model = load_model(os.path.abspath(modelPath  + '.h5'), custom_objects={"top1": top1,"top5": top5})

image_dim = 224 

print('\nInput shape: ' + str(image_dim)  + '\n')

# setting inputs
input_shape = (image_dim, image_dim, 3)

#image dimensions
print('\nInput shape: ' + str(input_shape)  + '\n')

#read and preprocessing the image
img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
if img.shape != (image_dim,image_dim):
    img = cv2.resize(img, (image_dim,image_dim))

img_array = np.expand_dims(img, axis=0)
img_array = input_preprocessing[args.model](img_array)

####################### Prediction
y_pred1 = model.predict(img_array, steps=1)[0]
#pred = np.argmax(y_pred1, axis=1)

#model response
res = getResponse(y_pred1, classes, TopK)
print('\nPrediction:\n'+ res)
