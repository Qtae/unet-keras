import argparse

import tensorflow as tf

from data_manager import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('test_dir', type=str, help='test dataset directory')
  parser.add_argument('model_path', type=str, help='trained model path')
  args = parser.parse_args()
  test_dir = args.test_dir
  testset_path_list = get_dataset_path_list(test_dir)
  test_images, test_labels = load_data(testset_path_list, (192, 192))
  
  #print(train_images.shape)
  #print(train_labels.shape)
  #print(test_images.shape)
  #print(test_labels.shape)

  #monitor_dataset(test_images, test_labels)

model = tf.keras.models.load_model(args.model_path)
model.summary()
res = model.predict(test_images)
monitor_dataset(test_images, res)