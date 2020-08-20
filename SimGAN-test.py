
import os
import h5py
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras import applications
from keras.layers.convolutional import UpSampling2D, MaxPooling2D, Deconv2D, Conv2D
from keras.layers import Input, Activation, Add
from keras.models import Sequential, Model
from utils.helper import plot_batch

def refiner_model(width = 55, height = 35, channels = 1):
	"""
	The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
	than holistically modifying the image content, preserving the global structure and annotations.
	
	:param input_image_tensor: Input tensor that corresponds to a synthetic image.
	:return: Output tensor that corresponds to a refined synthetic image.
	"""
	
	def resnet_block(input_features, nb_features=64, kernel_size=3):
		"""
		A ResNet block with two `kernel_size` x `kernel_size` convolutional layers,
		each with `nb_features` feature maps.
		
		See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
		
		:param input_features: Input tensor to ResNet block.
		:return: Output tensor from ResNet block.
		"""
		y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(input_features)
		y = Activation('relu')(y)
		y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(y)
		
		y = Add()([y, input_features])
		y = Activation('relu')(y)
		
		return y

	input_layer = Input(shape=(height, width, channels))
	# an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
	x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(input_layer)

	for _ in range(4):
		x = resnet_block(x)

	output_layer = Conv2D(channels, kernel_size=1, padding='same', activation='tanh')(x)

	return Model(input_layer, output_layer, name='refiner')



def get_image_batch(generator,batch_size):
	"""keras generators may generate an incomplete batch for the last batch"""
	img_batch = generator.next()
	if len(img_batch) != batch_size:
		img_batch = generator.next()
	
	assert len(img_batch) == batch_size
	return img_batch


if __name__ == '__main__':
	### load model
	model_path = os.path.join('.', 'cache', 'refiner_model.h5')
	refiner = refiner_model(width = 55, height = 35, channels = 1)
	refiner.load_weights(model_path)

	###  Data Generators  ###
		###  Loading the Data  ###
	path = os.path.dirname(os.path.abspath('.'))
	data_dir = os.path.join('.', 'dataset')
	output_dir = os.path.join('.','output')

	# load the data file and extract dimentions
	with h5py.File(os.path.join(data_dir, 'gaze.h5'), 'r') as t_file:
		syn_img_stack = np.stack([np.expand_dims(a, -1) for a in t_file['image'].values()], 0)

	test_size = 16
	datagen = image.ImageDataGenerator(preprocessing_function=applications.xception.preprocess_input, data_format='channels_last')
	syn_gen = datagen.flow(x=syn_img_stack, batch_size=test_size)

	syn_imgs = get_image_batch(syn_gen, test_size)
	refined_imgs = refiner.predict_on_batch(syn_imgs)

	figure_name = 'synImage_vs_refinedImg.png'

	plot_batch(np.concatenate((syn_imgs, refined_imgs)), os.path.join(output_dir, figure_name), 
                label_batch=['Synthetic'] * test_size + ['Refined'] * test_size)