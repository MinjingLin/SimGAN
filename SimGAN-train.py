import sys
import time
import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from itertools import groupby
from skimage.util import montage

from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate, Activation, Add
from keras.layers.convolutional import UpSampling2D, MaxPooling2D, Deconv2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras import initializers
from keras import applications
from keras.utils import plot_model
from keras.preprocessing import image

import tensorflow as tf
from utils.helper import plot_batch

# install pydot_ng,graphviz as follows first to plot model:
# pip install pydot_ng
# pip install graphviz
# sudo dot -c

from keras.utils import plot_model


# assert the tensorflow and keras version
# for me, tf version is:  2.2.0, keras version is:  2.4.3
print('tf version is: ', tf.__version__)
import keras
print('keras version is: ',keras.__version__)

img_width = 55
img_height = 35
channels = 1

batch_size = 512
learning_rate=0.001

def local_adversarial_loss(y_true, y_pred):
    truth = tf.reshape(y_true, (-1, 2))
    predicted = tf.reshape(y_pred, (-1, 2))
    
    computed_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth, logits=predicted)
    output = tf.reduce_mean(computed_loss)
    
    return output


def self_regularisation_loss(y_true, y_pred):
    return tf.multiply(0.0002, tf.reduce_sum(tf.abs(y_pred - y_true)))
# reduce_sum: Computes the sum of elements across dimensions of a tensor.

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


def discriminator_model(width = 55, height = 35, channels = 1):
    input_layer = Input(shape=(height, width, channels))

    x = Conv2D(96, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(2, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    output_layer=Reshape(target_shape=(x.shape[1]*x.shape[2],2))(x)
    #output_layer = Reshape(target_shape=(-1, 2))(x)

    return Model(input_layer, output_layer, name='discriminator')



####  Building the SimGAN  ####
refiner = refiner_model(img_width, img_height, channels)
refiner.compile(loss=self_regularisation_loss, optimizer=SGD(lr=learning_rate))
refiner.summary()
#plot_model(refiner,show_shapes=False,to_file='./output/refiner_model.png')

disc = discriminator_model(img_width, img_height, channels)
disc.compile(loss=local_adversarial_loss, optimizer=SGD(lr=learning_rate))
disc.summary()
#plot_model(disc,show_shapes=True,to_file='./output/discriminator_model.png')


synthetic_img = Input(shape=(img_height, img_width, channels))
refined_output = refiner(synthetic_img)
discriminator_output = disc(refined_output)
combined_model = Model(inputs=synthetic_img, outputs=[refined_output, discriminator_output], name='combined')
combined_model.summary()
#plot_model(combined_model,show_shapes=False,to_file='./output/simGAN_model.png')


disc.trainabler = False
combined_model.compile(loss=[self_regularisation_loss, local_adversarial_loss], optimizer=SGD(lr=learning_rate))


###  Loading the Data  ###
#path = os.path.dirname(os.path.abspath('.'))
data_dir = os.path.join('.', 'dataset')
cache_dir = os.path.join('.','cache')

# load the data file and extract dimentions
with h5py.File(os.path.join(data_dir, 'gaze.h5'), 'r') as t_file:
    syn_img_stack = np.stack([np.expand_dims(a, -1) for a in t_file['image'].values()], 0)
    
with h5py.File(os.path.join(data_dir, 'real_gaze.h5'), 'r') as t_file:
    real_img_stack = np.stack([np.expand_dims(a, -1) for a in t_file['image'].values()], 0)



###  Data Generators  ###
datagen = image.ImageDataGenerator(preprocessing_function=applications.xception.preprocess_input, data_format='channels_last')
syn_gen = datagen.flow(x=syn_img_stack, batch_size=batch_size)
real_gen = datagen.flow(x=real_img_stack, batch_size=batch_size)
def get_image_batch(generator):
    """keras generators may generate an incomplete batch for the last batch"""
    img_batch = generator.next()
    if len(img_batch) != batch_size:
        img_batch = generator.next()
    
    assert len(img_batch) == batch_size
    return img_batch

disc_output_shape = disc.output_shape

y_real_label = np.array([[[1.0, 0.0]] * disc_output_shape[1]] * batch_size)
y_refined_label = np.array([[[0.0, 1.0]] * disc_output_shape[1]] * batch_size)

assert y_real_label.shape == (batch_size, disc_output_shape[1], 2)
assert y_refined_label.shape == (batch_size, disc_output_shape[1], 2)

batch_out = get_image_batch(syn_gen)
assert batch_out.shape == (batch_size, img_height, img_width, channels), "Image dimension do not match, {} != {}" \
    .format(batch_out.shape, (batch_size, img_height, img_width, img_channels))


###  Pretraining  ###
plotted_imgs = 16
## Pretraining the Generator(Refiner) ##
def pretrain_gen(steps, log_interval, save_path, profiling=True):
    losses = []
    gen_loss = 0.
    if profiling:
        start = time.perf_counter()
    for i in range(steps):
        syn_imgs_batch = get_image_batch(syn_gen)
        loss = refiner.train_on_batch(syn_imgs_batch, syn_imgs_batch)
        gen_loss += loss

        if (i+1) % log_interval == 0:
            print('pre-training generator step {}/{}: loss = {:.5f}'.format(i+1, steps, gen_loss / log_interval))
            losses.append(gen_loss / log_interval)
            gen_loss = 0.
        
        if (i+1) % (5*log_interval) == 0:
            figure_name = 'refined_img_pretrain_step_{}.png'.format(i)
            syn_imgs = get_image_batch(syn_gen)[:plotted_imgs]
            gen_imgs = refiner.predict_on_batch(syn_imgs)

            plot_batch(np.concatenate((syn_imgs, gen_imgs)), os.path.join(cache_dir, figure_name), 
                       label_batch=['Synthetic'] * plotted_imgs + ['Refined'] * plotted_imgs)

    if profiling:
        duration = time.perf_counter() - start
        print('pre-training the refiner model for {} steps lasted = {:.2f} minutes = {:.2f} hours'.format(steps, duration / 60., duration / 3600.))

    #refiner.save(save_path)
    return losses

# we first train the Rθ network with just self-regularization loss for 1,000 steps
gen_pre_steps = 1000
gen_log_interval = 20

pre_gen_path = os.path.join(cache_dir, 'refiner_model_pre_trained_{}.h5'.format(gen_pre_steps))
if os.path.isfile(pre_gen_path):
    refiner.load_weights(pre_gen_path)
    print('loading pretrained model weights')
else:
    losses = pretrain_gen(gen_pre_steps, gen_log_interval, pre_gen_path)
    plt.plot(range(gen_log_interval, gen_pre_steps+1, gen_log_interval), losses)


## Pretraining the Discriminator ##
def pretrain_disc(steps, log_interval, save_path, profiling=True):
    losses = []
    disc_loss = 0.
    if profiling:
        start = time.perf_counter()
    for i in range(steps):
        real_imgs_batch = get_image_batch(real_gen)
        disc_real_loss = disc.train_on_batch(real_imgs_batch, y_real_label)
        
        syn_imgs_batch = get_image_batch(syn_gen)
        disc_refined_loss = disc.train_on_batch(syn_imgs_batch, y_refined_label)
        
        disc_loss += 0.5 * np.add(disc_real_loss, disc_refined_loss)

        if (i+1) % log_interval == 0:
            print('pre-training discriminator step {}/{}: loss = {:.5f}'.format(i+1, steps, disc_loss / log_interval))
            losses.append(disc_loss / log_interval)
            disc_loss = 0.

    if profiling:
        duration = time.perf_counter() - start
        print('pre-training the discriminator model for {} steps lasted = {:.2f} minutes = {:.2f} hours'.format(steps, duration/60., duration/3600.))
    
    #disc.save(save_path)
    return losses

# and Dφ for 200 steps (one mini-batch for refined images, another for real)
disc_pre_steps = 200
disc_log_interval = 20

pre_disc_path = os.path.join(cache_dir, 'disc_model_pre_trained_{}.h5'.format(disc_pre_steps))

if os.path.isfile(pre_disc_path):
    print('loading pretrained model weights')
    disc.load_weights(pre_disc_path)
else:
    losses = pretrain_disc(disc_pre_steps, disc_log_interval, pre_disc_path)
    plt.plot(range(disc_log_interval, disc_pre_steps+1, disc_log_interval), losses)



### Full Training ###
ihb = ImageHistoryBuffer((0, img_height, img_width, channels), batch_size*100, batch_size)

gan_loss = np.zeros(shape=len(combined_model.metrics_names))
disc_loss_real = 0.
disc_loss_refined = 0.
disc_loss = 0.
nb_steps = 2000 # originally 10000
k_d = 1 # number of discriminator updates per step
k_g = 2 # number of generator updates per step
log_interval = 40
# see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
for i in range(nb_steps):    
    # train the refiner
    for _ in range(k_g * 2):
        # sample a mini-batch of synthetic images
        syn_img_batch = get_image_batch(syn_gen)
        # update θ by taking an SGD step on mini-batch loss LR(θ)
        loss = combined_model.train_on_batch(syn_img_batch, [syn_img_batch, y_real_label])
        gan_loss = np.add(gan_loss, loss)
    
    for _ in range(k_d):
        # sample a mini-batch of synthetic and real images
        syn_img_batch = get_image_batch(syn_gen)
        real_img_batch = get_image_batch(real_gen)
        
        # refine the synthetic images w/ the current refiner
        refined_img_batch = refiner.predict_on_batch(syn_img_batch)
        
        # use a history of refined images
        history_img_half_batch = ihb.get_from_image_history_buffer()
        ihb.add_to_history_img_buffer(refined_img_batch)
        
        if len(history_img_half_batch):
            refined_img_batch[:batch_size//2] = history_img_half_batch
        
        # update φ by taking an SGD step on mini-batch loss LD(φ)
        real_loss = disc.train_on_batch(real_img_batch, y_real_label)
        disc_loss_real += real_loss
        ref_loss = disc.train_on_batch(refined_img_batch, y_refined_label)
        disc_loss_refined += ref_loss
        disc_loss += 0.5 * (real_loss + ref_loss)
    
    if (i+1) % log_interval == 0:
        print('step: {}/{} | [D loss: (real) {:.5f} / (refined) {:.5f} / (combined) {:.5f}]'.format(i+1, 
                      nb_steps, disc_loss_real/log_interval, disc_loss_refined/log_interval,  disc_loss/log_interval))
        
        gan_loss = np.zeros(shape=len(combined_model.metrics_names))
        disc_loss_real = 0.
        disc_loss_refined = 0.
        disc_loss = 0.
    
    if (i+1) % (log_interval*5) == 0:
        figure_name = 'refined_image_batch_step_{}.png'.format(i)
        print('Saving batch of refined images at adversarial step: {}.'.format(i))
        
        synthetic_image_batch = get_image_batch(syn_gen)[:plotted_imgs]
        plot_batch(
            np.concatenate((synthetic_image_batch, refiner.predict_on_batch(synthetic_image_batch))),
            os.path.join(cache_dir, figure_name),
            label_batch=['Synthetic']*plotted_imgs + ['Refined']*plotted_imgs)

refiner.save(os.path.join(cache_dir, 'refiner_model_{}.h5'.format(nb_steps)))
disc.save(os.path.join(cache_dir, 'disc_model_{}.h5'.format(nb_steps)))
combined_model.save(os.path.join(cache_dir, 'simgan_model_{}.h5'.format(nb_steps)))


