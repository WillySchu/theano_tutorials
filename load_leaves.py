from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys

import numpy

import theano
import theano.tensor as T

from PIL import Image

def load():
    species_list = [
        'test1',
        'test2',
        'test3'
    ]
    STANDARD_SIZE = (28, 28)

    def load_leaf_train(species):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'lab',
            species
        )


        directory = os.listdir(path)
        images = []
        labels = []
        for file in directory:
            if file != '.DS_Store':
                img = Image.open(path + '/' + file).convert('L')
                img = img.resize(STANDARD_SIZE)
                img = list(img.getdata())
                # img = map(list, img)
                img = numpy.array(img)
                images.append(img)
                labels.append(0)


        return [images, labels]

    def load_leaf_test(species):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'field',
            species
        )

        directory = os.listdir(path)
        images = []
        labels = []

        i = 0

        for file in directory:
            i += 1
            if file != '.DS_Store':
                if len(directory) // 2 > i:
                    img = Image.open(path + '/' + file).convert('L')
                    # img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append(img)
                    labels.append(0)


        return [images, labels]


    def load_leaf_valid(species):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'field',
            species
        )

        directory = os.listdir(path)
        images = []
        labels = []

        i = 0

        for file in directory:
            i += 1
            if file != '.DS_Store':
                if len(directory) // 2 < i:
                    img = Image.open(path + '/' + file).convert('L')
                    img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append(img)
                    labels.append(0)

        return [images, labels]

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        print(data_x.shape)
        print(data_y.shape)

        shared_x = theano.shared(numpy.asarray(data_x,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )
        shared_y = theano.shared(numpy.asarray(data_y,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )

        return shared_x, T.cast(shared_y, 'int32')

    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    images_valid = []
    labels_valid = []
    for species in species_list:
        print('... loading %s' % species)
        x_train, y_train = load_leaf_train(species)
        x_valid, y_valid = load_leaf_valid(species)
        x_test, y_test = load_leaf_test(species)

        images_train += x_train
        labels_train += y_train
        images_test += x_test
        labels_test += y_test
        images_valid += x_valid
        labels_valid += y_valid

    images_train = numpy.asarray(images_train)
    labels_train = numpy.asarray(labels_train)
    images_test = numpy.asarray(images_test)
    labels_test = numpy.asarray(labels_test)
    images_valid = numpy.asarray(images_valid)
    labels_valid = numpy.asarray(labels_valid)

    xy_test = [images_test, labels_test]
    xy_train = [images_train, labels_train]
    xy_valid = [images_valid, labels_valid]

    train_x, train_y = shared_dataset(xy_train)
    test_x, test_y = shared_dataset(xy_test)
    valid_x, valid_y = shared_dataset(xy_valid)

    rval = [(train_x, train_y), (test_x, test_y), (valid_x, valid_y)]

    return rval





def load_mnist(dataset='mnist.pkl.gz'):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == '' and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        print(data_x.shape)
        print(data_x[0].shape)

        shared_x = theano.shared(numpy.asarray(data_x,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )
        shared_y = theano.shared(numpy.asarray(data_y,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
