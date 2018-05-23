from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.datasets import cifar10
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import toimage

image_height = 32
image_width = 32
num_classes=10

def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(toimage(X[k]))
            k = k+1
    plt.show()

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()

train_features = train_features.astype(float)/255
test_features = test_features.astype(float)/255
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

model = load_model('cifar10_model_v2.h5')

input_dir = 'data/input/'
input_images = [input_dir + i for i in listdir(input_dir)]
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for file in input_images:
    image_to_classify = load_img(file, target_size=(image_height, image_width))
    image_to_classify = img_to_array(image_to_classify)
    image_to_classify = np.expand_dims(image_to_classify, axis=0)
    prediction = np.argmax(model.predict(image_to_classify), 1)
    print([labels[x] for x in prediction])


score = model.evaluate(test_features, test_labels, batch_size=128, verbose=0)
print('\nTest result: %.3f loss: %.3f' % (score[1]*100, score[0]))