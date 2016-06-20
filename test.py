import os
import numpy

from PIL import Image

STANDARD_SIZE = (128, 128)

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
    print(path)


    directory = os.listdir(path)

    i = 0
    for file in directory:
        if file != '.DS_Store':
            if i == 4: break
            img = Image.open(path + '/' + file).convert('L')
            img = img.resize(STANDARD_SIZE)
            img.save('test' + str(i) + '.jpg')

            i += 1

def crop(species):
    path = os.path.join(
        os.path.split(__file__)[0],
        'data',
        'leafsnap-dataset',
        'dataset',
        'images',
        'lab',
        species
    )
    i = 0
    directory = os.listdir(path)
    for file in directory:
        if file != '.DS_Store':
            if i == 5: break
            img = Image.open(path + '/' + file).convert('L')
            img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
            img.show()

            i += 1

if __name__=='__main__':
    crop('acer_rubrum')
