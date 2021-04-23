import os
import time

import tensorflow as tf

from model import unet
from data_manager import *


if __name__ == '__main__':
  ## load data ###########################################################
  train_dir = 'D:\\Work\\04_PGM\\Data\\ScratchSegmentationData\\0_Train'
  trainset_path_list = get_dataset_path_list(train_dir)
  train_images, train_labels, valid_images, valid_labels = load_data(trainset_path_list, (192, 192), 0.2)
  
  print(train_images.shape)
  print(train_labels.shape)
  print(valid_images.shape)
  print(valid_labels.shape)

  ## build network #######################################################
  input = tf.keras.layers.Input((192, 192, 3))

  model = unet(input)

  adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  model.compile(optimizer=adam,
                loss=bce,
                metrics=[tf.keras.metrics.MeanIoU(num_classes=2), 'accuracy'])
  model.summary()

  ## callback functions ########################################
  savedir = os.path.join("D:\\Work\\04_PGM\\unet\\checkpoints", time.strftime('model_%Y%m%d%H%M%S', time.localtime(time.time())))
  if not os.path.exists(savedir):
    os.makedirs(savedir)
  filename = "SEG_e{epoch:02d}-acc{val_accuracy:.3f}.hdf5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(savedir, filename),
                                                  verbose=1,  # 1 = progress bar
                                                  monitor='val_accuracy',
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='auto')

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='auto')
  reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.5,
                                                patience=3,
                                                cooldown=10,
                                                min_lr=0.000001,
                                                mode='auto')

  callbacks_list = [checkpoint, early_stopping, reduce]
  
  ## train ####################################################
  history = model.fit(x=train_images,
                      y= train_labels,
                      validation_data=(valid_images,valid_labels),
                      batch_size=8,
                      epochs=500,
                      callbacks=callbacks_list)
