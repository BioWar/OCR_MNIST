import os
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm
from pandas import DataFrame
from random import random, randint
from argparse import ArgumentParser

def generate_image_label(amount, path_to_images, path_to_labels):
  os.makedirs(path_to_images, exist_ok=True)
  df = DataFrame(columns=['filename', 'gt'])
  labels = []
  for i_img in tqdm(range(amount)):
    canvas = np.zeros(shape=(28, 28*10))
    string = ""
    for i in range(10):
      if random() < 0.5:
        idx = randint(0, x_train.shape[0]-1)
        canvas[:, i*28:i*28+28] = np.squeeze(x_train[idx])
        y_label = y_train[idx]
        char = "s_" + str(y_label)
      else:
        char = '|'
      string += char + '-'
    string = string[:-1]
    labels.append(string)
    im = Image.fromarray(canvas*255)
    im = im.convert("L")
    filename = i_img
    im.save(os.path.join(path_to_images, f"{filename}.jpeg"))
    df.loc[i_img] = [filename, string]
  df.to_csv(os.path.join(path_to_labels, "transcription.csv"), index=False) 
  print()
  # print(os.path.join(path_to_labels, "transcription.csv"))
  return df

if __name__=="__main__":
  parser = ArgumentParser(description="MNIST HWR dataset generation")
  parser.add_argument("-n", required=True, type=int,
                      help="amount of images to generate")
  parser.add_argument("-i", "--images", required=True, type=str,
                      help="path to directory where images will be saved")
  parser.add_argument("-t", "--transcription", required=True, type=str,
                      help="path to directory where transcription file will be saved")
  args = parser.parse_args()

  print("\nLoad MNIST dataset...")
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Rescale the images from [0,255] to the [0.0,1.0] range.
  x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

  print("Number of original training examples:", len(x_train))
  print("Number of original test examples:", len(x_test))

  amount = args.n
  path_to_images = args.images
  path_to_labels = args.transcription

  print(f"Path to images: {path_to_images}")
  print(f"Path to labels: {path_to_labels}")
  print(f"Images will be generated: {amount}")
  print(f"Vocabulary: '0123456789 '. Additional symbol '-' for empty space between chars.")
  print("Encoding: {' ': '|', '': '-', '1': 's_1', '2': 's_2', ...}")

  df = generate_image_label(amount, path_to_images, path_to_labels)
