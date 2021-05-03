import tensorflow as tf

from data_manager import *

if __name__ == '__main__':
  test_dir = 'D:\\Work\\04_PGM\\Data\\ScratchSegmentationData\\0_Test'
  testset_path_list = get_dataset_path_list(test_dir)
  test_images, test_labels = load_data(testset_path_list, (192, 192))
  
  #print(train_images.shape)
  #print(train_labels.shape)
  #print(test_images.shape)
  #print(test_labels.shape)

  #monitor_dataset(test_images, test_labels)

model = tf.keras.models.load_model("D:\\Work\\04_PGM\\unet\\210406_model1.hdf5")
model.summary()
res = model.predict(test_images)
monitor_dataset(test_images, res)