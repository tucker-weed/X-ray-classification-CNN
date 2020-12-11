from __future__ import absolute_import
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import datetime
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        Architechture for CNN
        """
        super(Model, self).__init__()

        self.batch_size = 20
        self.num_classes = 2

        # Initialize all hyperparameters
        self.learning_rate = 0.0001
        self.num_epochs = 3

        # Hyperparameters dealing with how data is read from memory

        # Indicates whether to read segments of the entire dataset
        self.using_all_data = True
        # Stop after segment. Assign -1 to allow all data segments to be read
        self.stop_at_segment = 3

        # If using_all_data is False, one segment will be read for train and test
        # These hyperparameters control their segment size in this case
        self.partial_data_train_segment = 320
        self.partial_data_test_segment = 120
        

        # Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([10,10,4,200], stddev=.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([10,10,200,100], stddev=.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([10,10,100,100], stddev=.1))
        self.filter4 = tf.Variable(tf.random.truncated_normal([10,10,100,50], stddev=.1))

        self.D1 = tf.keras.layers.Dense(self.batch_size, activation="relu")
        self.D2 = tf.keras.layers.Dense(self.batch_size, activation="relu")
        self.D3 = tf.keras.layers.Dense(self.num_classes)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        layer1Output = tf.nn.conv2d(inputs, self.filter1, strides=5, padding='SAME')
        (mean1, variance1) = tf.nn.moments(layer1Output, [0,1,2])
        layer1Output = tf.nn.batch_normalization(layer1Output, mean1, variance1, variance_epsilon=0.00000001, offset=None, scale=None)
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.max_pool(layer1Output, 3, 2, padding='SAME')

        layer2Output = tf.nn.conv2d(layer1Output, self.filter2, strides=4, padding='SAME')
        (mean2, variance2) = tf.nn.moments(layer2Output, [0,1,2])
        tf.nn.batch_normalization(layer2Output, mean2, variance2, variance_epsilon=0.00000001, offset=None, scale=None)
        layer2Output = tf.nn.relu(layer2Output)
        layer2Output = tf.nn.max_pool(layer2Output, 2, 2, padding='SAME')

        layer3Output = tf.nn.conv2d(layer2Output, self.filter3, strides=4, padding='SAME')
        (mean5, variance5) = tf.nn.moments(layer3Output, [0,1,2])
        tf.nn.batch_normalization(layer3Output, mean5, variance5, variance_epsilon=0.00000001, offset=None, scale=None)
        layer3Output = tf.nn.relu(layer3Output)

        layer3Output = tf.nn.conv2d(layer3Output, self.filter4, strides=3, padding='SAME')
        (mean5, variance5) = tf.nn.moments(layer3Output, [0,1,2])
        tf.nn.batch_normalization(layer3Output, mean5, variance5, variance_epsilon=0.00000001, offset=None, scale=None)
        layer3Output = tf.nn.relu(layer3Output)

        layer3Output = tf.reshape(layer3Output, [len(inputs), -1])
        layer3Output = self.D1(layer3Output)
        if not is_testing:
            tf.nn.dropout(layer3Output, rate=0.3)
        layer3Output = self.D2(layer3Output)
        if not is_testing:
            tf.nn.dropout(layer3Output, rate=0.3)
        logits = self.D3(layer3Output)
    
        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: None
    '''
    BATCH_SZ = model.batch_size

    inds = np.arange(0, np.shape(train_inputs)[0])
    np.random.shuffle(inds)
    train_inputs = train_inputs[inds]
    train_labels = train_labels[inds]
    
    steps = 0
    for i in range(0, np.shape(train_inputs)[0], BATCH_SZ):
        steps += 1
        image = train_inputs[i:i + BATCH_SZ]
        label = train_labels[i:i + BATCH_SZ]
        with tf.GradientTape() as tape:
            predictions = model.call(image)
            loss = model.loss(predictions, label)

        train_acc = model.accuracy(model(image), label)
        print("Accuracy on training set after {} training steps: {}".format(steps, train_acc))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels, setType):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :param setType: validation or test string
    :return: test accuracy
    """
    BATCH_SZ = model.batch_size
    accs = []

    inds = np.arange(0, np.shape(test_inputs)[0])
    np.random.shuffle(inds)
    test_inputs = test_inputs[inds]
    test_labels = test_labels[inds]

    steps = 0
    for i in range(0, np.shape(test_inputs)[0], BATCH_SZ):
        steps += 1
        image = test_inputs[i:i + BATCH_SZ]
        label = test_labels[i:i + BATCH_SZ]
        predictions = model.call(image, is_testing=True)
        acc = model.accuracy(predictions, label)
        print("Accuracy on {} set after {} training steps: {}".format(setType, steps, acc))
        accs.append(acc)
    return tf.reduce_mean(tf.convert_to_tensor(accs))


def main():
    '''
    Executes training and testing steps
    
    :return: None
    '''
    model = Model()

    pn = 0
    pp = 0
    for epoch in range(model.num_epochs):
        last = 0.0
        curr = 0.0
        mem_seg = 0
        end = False
        while not end:
            mem_seg += 1
            (inp_train, lab_train, pn, pp, end) = None, None, pn, pp, None
            if not model.using_all_data:
                (inp_train, lab_train, pn, pp, end) = get_data("../data/train", positionN=pn, positionP=pp)
            else:
                (inp_train, lab_train, pn, pp, end) = get_data("../data/train", 
                                                            segment=model.partial_data_train_segment, positionN=pn, positionP=pp)
            print("\nTRAIN DATA SEGMENT: {} | EPOCH: {}\n".format(mem_seg, epoch + 1))
            train(model, inp_train, lab_train)
            if not model.using_all_data or (not model.stop_at_segment == -1 and mem_seg == model.stop_at_segment):
                end = True
        if epoch < model.num_epochs - 1:
                print("\nVALIDATION TEST\n")
                (inp_val, lab_val, _a, _b, _c) = get_data("../data/val", segment=16)
                curr = test(model, inp_val, lab_val, setType="validation")
                if curr < last:
                    print("\nEARLY STOP\n")
                    break
                else:
                    last = curr

    inp_train = None
    lab_train = None

    print("\nTEST SET\n")
    pn = 0
    pp = 0
    mem_seg = 0
    end = False
    test_accuracies = []
    if model.stop_at_segment > 1:
        model.stop_at_segment -= 1
    while not end:
        mem_seg += 1
        (inp_test, lab_test, pn, pp, end) = None, None, pn, pp, None
        if model.using_all_data:
            (inp_test, lab_test, pn, pp, end) = get_data("../data/test", positionN=pn, positionP=pp)
        else:
            (inp_test, lab_test, pn, pp, end) = get_data("../data/test", 
                                                         segment=model.partial_data_test_segment, positionN=pn, positionP=pp)
        print("\nTEST DATA SEGMENT: {}\n".format(mem_seg))
        test_accuracies.append(test(model, inp_test, lab_test, setType="test"))
        if not model.using_all_data or (not model.stop_at_segment == -1 and mem_seg == model.stop_at_segment):
            end = True

    print("\nFINAL TEST ACCURACY: {}\n".format(tf.reduce_mean(tf.convert_to_tensor(test_accuracies))))


if __name__ == '__main__':
    main()
