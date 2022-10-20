# Prepare the data using fer2013.csv and fer2013new.csv

import os
import csv
import pandas as pd
import numpy as np
from PIL import Image

DATA_FOLDER = 'data'
IMAGE_FOLDER = 'images'
FER2013 = 'fer2013.csv'
FER2013NEW = 'fer2013new.csv'
FER2013NEW_CLEAR = 'fer2013new_clear.csv'
IMAGE_SIZE = (48, 48)


def prepare_data(data_folder):
    """ Extract images from fer2013.csv and name them as specified in fer2013new.csv.
    Unnamed or unlabeled images are discarded.
    Create a new fer2013new_clear.csv that contains only valid rows.
    """
    folder = os.path.join(data_folder, IMAGE_FOLDER)
    if not os.path.exists(folder):
        os.makedirs(folder)
    fer2013 = csv.DictReader(open(os.path.join(data_folder, FER2013)), delimiter=',')
    fer2013new = pd.read_csv(os.path.join(data_folder, FER2013NEW))
    i = 0
    for i, row in enumerate(fer2013):
        filename = str(fer2013new['Image name'][i]).strip()
        unlabeled = fer2013new['NF'][i] + fer2013new['unknown'][i]
        if filename and unlabeled < 10:
            image_string = row['pixels'].split(' ')
            image = Image.fromarray(np.asarray(image_string, dtype=np.uint8).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1]))
            image.save(os.path.join(folder, filename), compress_level=0)
    print(f"{i+1} FER+ dataset images extracted to '{folder}'")
    fer2013new_clear = fer2013new.dropna()
    fer2013new_clear = fer2013new_clear[(fer2013new_clear['NF'] + fer2013new_clear['unknown']) < 10]
    fer2013new_clear.to_csv(os.path.join(data_folder, FER2013NEW_CLEAR))
    print(f"Cleared dataset labels saved to {os.path.join(data_folder, FER2013NEW_CLEAR)}")


if __name__ == '__main__':
    prepare_data(DATA_FOLDER)
