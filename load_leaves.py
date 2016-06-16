from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys

import numpy

import theano
import theano.tensor as T

from PIL import Image

def load_leaf():
    path = os.path.join(
        os.path.split(__file__)[0],
        'data',
        'leafsnap-dataset',
        'dataset',
        'images',
        'field',
        'abies_concolor',
        '12995307070714.jpg'
    )
    print(path)

    img = Image.open(path)
    # img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = numpy.array(img)
    return img

    # img = Image.open(path).convert('L')
    # img.load()
    #
    # data = numpy.asarray(img, dtype = theano.config.floatX)
    #
    # return data






def load_mnist(dataset):
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
