# Emotion detection from the webcam feed or on an image

import os
import argparse
from model import *
from train import MODEL_FILE
from PIL import Image, ImageOps
from dataset import EMOTIONS, resnet_transform
from torch.nn.functional import softmax
import cv2 as cv
import numpy as np


FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMOJI_FOLDER = 'emoji'
EMOJI_SIZE = 36  # size of 1 emoji in webcam output
EMOTION_THRESHOLD = 0.10  # Minimum score for emoji to be shown


def make_emoji_line(emoji_images, emotions):
    """ Creates image with emojis with transparency corresponding to emotion scores."""
    base_size = EMOJI_SIZE
    emotions = {emotion: score for emotion, score in emotions.items() if score > EMOTION_THRESHOLD}
    image = np.zeros((base_size, base_size*len(emotions), 4), dtype=np.float32)
    image[:, :, 3] = 1.0
    for i, emotion in enumerate(sorted(emotions, key=lambda i: emotions[i], reverse=True)):
        emoji = emoji_images[emotion]
        score = emotions[emotion]
        image[:, i * base_size:(i + 1) * base_size, :] = emoji
        alpha_channel = image[:, i * base_size:(i + 1) * base_size, 3]
        image[:, i * base_size:(i + 1) * base_size, 3][alpha_channel > 0] = 1.0 - score
        image[:, i * base_size:(i + 1) * base_size, 3][alpha_channel <= 0] = 1.0
    return image


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', default='webcam', choices=['webcam', 'image'],
                           help='webcam to get real-time input from webcam, image to classify one image')
    argparser.add_argument('--imgpath', type=str,
                           help='path to image in case of image classification mode')
    args = argparser.parse_args()
    mode = args.mode
    img_path = args.imgpath

    model = ResNet()
    model.load_state_dict(torch.load(MODEL_FILE))

    if mode == 'image':
        assert os.path.exists(img_path)
        image = Image.open(img_path)
        image = ImageOps.grayscale(image)
        image = resnet_transform(image)
        prediction = softmax(model(image[None, :]), dim=-1)
        print(EMOTIONS)
        print(prediction)

    elif mode == 'webcam':
        # Use OpenCV pre-trained cascade classifier for face recognition
        # https://github.com/opencv/opencv/tree/3.4/data/haarcascades
        face_cascade = cv.CascadeClassifier()

        emoji_images = {emotion: cv.resize(cv.imread(os.path.join(EMOJI_FOLDER, emotion+'.png'),
                                           cv.IMREAD_UNCHANGED), (EMOJI_SIZE, EMOJI_SIZE))
                        for emotion in EMOTIONS if emotion != 'neutral'}
        if not face_cascade.load(FACE_CASCADE_PATH):
            raise Exception(f'Failed to load {FACE_CASCADE_PATH} , make sure it is in the program directory.')

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise Exception('Error opening video capture.')

        while True:
            ret, frame = cap.read()
            if frame is None:
                print("No captured frame - Terminating")
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray)

            for (x, y, w, h) in faces:
                faceROI = frame_gray[y:y + h, x:x + w]
                image = Image.fromarray(faceROI)
                image = resnet_transform(image)
                prediction = softmax(model(image[None, :]), dim=-1)

                emotions = dict(zip(EMOTIONS, prediction.tolist()[0]))
                if 'neutral' in emotions:
                    del emotions['neutral']
                # Show all emotions as emoji in a row, set transparency proportionate to predicted score
                emoji_line = make_emoji_line(emoji_images, emotions)
                e_w = emoji_line.shape[1]
                e_h = emoji_line.shape[0]
                center_x = x + w // 2
                center_x = max(e_w//2, center_x)
                center_x = min(frame.shape[1]-1-e_w//2, center_x)
                y_bottom = min(y + h + 10 + e_h, frame.shape[0]-1)
                alpha = emoji_line[:, :, 3]
                for color in range(3):
                    frame[y_bottom-e_h:y_bottom, center_x-e_w//2:center_x+e_w//2, color] = (1 - alpha) * emoji_line[:, :, color] + \
                        frame[y_bottom-e_h:y_bottom, center_x-e_w//2:center_x+e_w//2, color] * alpha

            cv.imshow('Emotion detector', frame)
            if cv.waitKey(10) == 27:  # Exit on ESC
                break
        cap.release()
        cv.destroyAllWindows()
