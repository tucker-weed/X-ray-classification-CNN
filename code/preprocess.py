import pickle
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import random
import datetime


def get_data(prefix, mode="train"):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""

	random.seed(datetime.time())
	train_stop = 0
	test_stop = 0
	val_stop = 0
	if mode == "train":
		NUM_INPUTS = 300
		train_stop = NUM_INPUTS // 2
	elif mode == "test":
		NUM_INPUTS = 140
		test_stop = NUM_INPUTS // 2
	elif mode == "validation":
		NUM_INPUTS = 80
		val_stop = NUM_INPUTS // 2
	images = np.zeros((NUM_INPUTS, 1200, 1200, 4))
	labels = np.zeros((NUM_INPUTS, 2))
	idx = 0
	missed_idxs = []

	i = 0
	# NORMAL subset
	for filename in os.listdir(prefix + "/NORMAL"):
		i += 1
		if (i == val_stop and mode == "validation") or (i == test_stop and mode == "test") or i == train_stop:
			break
		if filename.endswith(".jpeg"): 
			image = Image.open(prefix + "/NORMAL/" + filename)
			image = image.resize((2400, 2400))
			image = np.array(image).astype('float32')  / 255.0
			image = np.reshape(image, (-1, 4, 1200, 1200))
			image = np.transpose(image, axes=[0,2,3,1])
			labels[idx] = np.array([0, 1]).astype('float32')
			images[idx] = image
		if random.randint(0, 1) == 1:
			missed_idxs.append(idx + 1)
			idx += 2
		else:
			idx += 1

	i = 0
	# PNEUMONIA subset
	for filename in os.listdir(prefix + "/PNEUMONIA"):
		i += 1
		if (i == val_stop and mode == "validation") or (i == test_stop and mode == "test") or i == train_stop:
			break
		if filename.endswith(".jpeg"): 
			image = Image.open(prefix + "/PNEUMONIA/" + filename)
			image = image.resize((2400, 2400))
			image = np.array(image).astype('float32') / 255.0
			image = np.reshape(image, (-1, 4, 1200, 1200))
			image = np.transpose(image, axes=[0,2,3,1])

			# For some reason a single image ends up size 3 on axis=0
			# The check below essentially turns it back to size 1 on axis=0
			if np.shape(image)[0]  == 3:
				image = image[0]
			if len(missed_idxs) > 0:
				newIdx = missed_idxs.pop(0)
				labels[newIdx] = np.array([1, 0]).astype('float32')
				images[newIdx] = image
			else:
				labels[idx] = np.array([1, 0]).astype('float32')
				images[idx] = image
				idx += 1

	return images, labels

