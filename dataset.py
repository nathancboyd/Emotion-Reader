# Custom dataset and transform definitions

import os
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from prepare_data import DATA_FOLDER, FER2013NEW_CLEAR, IMAGE_FOLDER

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

resnet_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.ToTensor(),
])


def labels_rescale(x):
    return x/x.sum()


class FERplusDataset(Dataset):
    def __init__(self, type='full', transform=resnet_transform, target_transform=labels_rescale):
        self.fer2013new = pd.read_csv(os.path.join(DATA_FOLDER, FER2013NEW_CLEAR))
        if type == 'train':
            self.fer2013new = self.fer2013new[self.fer2013new['Usage'] == 'Training']
        elif type == 'val':
            self.fer2013new = self.fer2013new[self.fer2013new['Usage'] != 'Training']
        self.fer2013new = self.fer2013new.reset_index()
        self.img_dir = os.path.join(DATA_FOLDER, IMAGE_FOLDER)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fer2013new)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.fer2013new['Image name'][idx])
        image = Image.open(img_path)
        labels = self.fer2013new[EMOTIONS].iloc[idx].to_numpy()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels
