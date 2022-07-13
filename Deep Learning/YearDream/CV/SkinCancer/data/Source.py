import tensorflow as tf
import numpy as np
import keras
import cv2
#import scikitplot as skplt
#import matplotlib.pyplot as plt


from keras import models, layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import keras.backend as K
from keras.optimizers import SGD
from keras.models import Model


batch_size = 128
epochs = 300
LearningRate = 1e-3
Decay = 1e-6
batch_size = 32
img_width = 224
img_height = 224

CurrentDirectory = "C:/Users/LeeJH/Downloads/Python/"

train_directory = CurrentDirectory + 'TRAIN/'
test_directory	= CurrentDirectory + 'TEST/'
model_directory = CurrentDirectory + 'MODEL/'
tensorboard_directory = CurrentDirectory + 'Tensorboard'


def VGG_16():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	#top layer of the VGG net
	return model


vggModel = VGG_16()
x = GlobalAveragePooling2D()(vggModel.output)
predictions = Dense(2, activation='softmax')(x)
DeepLearning = Model(inputs=vggModel.input, outputs=predictions)

DeepLearning.compile(optimizer=SGD(lr=LearningRate,decay=Decay,
	momentum=0.9,nesterov=True),loss='categorical_crossentropy',metrics=['acc'])


DATAGEN = ImageDataGenerator(
	rescale=1./255,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2, 
	horizontal_flip=True,
	vertical_flip=True,
	featurewise_center=True,
	featurewise_std_normalization=True,
	data_format="channels_last")

DATAGEN_TEST = ImageDataGenerator(
	rescale=1./255,
	featurewise_center=True,
	featurewise_std_normalization=True,
	data_format="channels_last")

TRAIN_GENERATOR = DATAGEN.flow_from_directory(
	train_directory,
	target_size = (img_width, img_height),
	batch_size = batch_size,
	class_mode='categorical')

TEST_GENERATOR = DATAGEN_TEST.flow_from_directory(
	test_directory,
	target_size = (img_width, img_height),
	batch_size = batch_size,
	shuffle = False,
	class_mode='categorical')


CP = ModelCheckpoint(filepath=model_directory+
					'-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.hdf5',
					monitor='val_acc', verbose=1, save_best_only=True, mode='max')
TB = TensorBoard(log_dir=tensorboard_directory, write_graph=True, write_images=True)
LR = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=3, verbose=1, min_lr=1e-8)
CALLBACK = [CP, TB, LR]


########## Training Start
DeepLearning.fit_generator(
	TRAIN_GENERATOR,
	steps_per_epoch=3,
	epochs=200,
	callbacks=CALLBACK,
	shuffle=True,
	validation_data=TEST_GENERATOR,
	validation_steps=1)

###########
	
DeepLearning.load_weights(model_directory+'PretrainedVGG.hdf5')
test_pred=DeepLearning.predict_generator(TEST_GENERATOR,verbose=1, steps=2)





#pip install scikit-plot, matplotlib
#pip install matplotlib

import scikitplot as skplt
import matplotlib.pyplot as plt


Labels = np.array([0, 1])
y_true = np.repeat(Labels, [30, 30], axis=0)
pred = test_pred[:,0]

skplt.metrics.plot_roc_curve(y_true, test_pred)
plt.show()



##### Heatmap

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pandas as pd

Files = ["ISIC_0028335.jpg", "ISIC_0028659.jpg", "ISIC_0028872.jpg", "ISIC_0029043.jpg"]
for i in range(len(Files)):
	img_path = CurrentDirectory+"Test/akiec/"+Files[i]
	output_path = CurrentDirectory + 'Heatmap_' + Files[i]

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	x = preprocess_input(x)
	preds = DeepLearning.predict(x)

	argmax = np.argmax(preds[0])
	output = DeepLearning.output[:, argmax]
	last_conv_layer = DeepLearning.get_layer('conv2d_13')

	grads = K.gradients(output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))
	iterate = K.function([DeepLearning.input], [pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([x])

	for i in range(512):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)

	img = cv2.imread(img_path)
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	hif = .8
	superimposed_img = heatmap * hif+ img

	cv2.imwrite(output_path, superimposed_img)






























































