import os
import random
import numpy as np
import json
import cv2

def shuffle_2Dlist(_input):
  if (_input is None) or (len(_input[0]) == 0):
    return [[],[]]
  c= list(zip(_input[0],_input[1]))
  random.shuffle(c)
  a,b = zip(*c)

  return [list(a),list(b)]

def get_dataset_path_list(dir):
  img_list = []
  json_list = []
  for filename in os.listdir(dir):
    file_ext = filename.split('.')[-1]
    if (file_ext == 'jpeg'):
      img_list.append(os.path.join(dir, filename))
      if os.path.exists(os.path.join(dir, filename.replace('.jpeg','.json'))):
        json_list.append(os.path.join(dir, filename.replace('.jpeg','.json')))
      else:
        json_list.append('no_json_file')
  dataset_path_list = [img_list, json_list]
  dataset_path_list = shuffle_2Dlist(dataset_path_list)

  return dataset_path_list

def load_data(dataset_path_list, img_size, valid_split):
  img_path_list = dataset_path_list[0]
  json_path_list = dataset_path_list[1]
  flag_list = []

  img_list_l = []
  label_list_l = []
  img_list_n = []
  label_list_n = []

  for img_filepath, json_filepath in zip(img_path_list, json_path_list):
    if (json_filepath == 'no_json_file'):
      img = cv2.resize(cv2.imread(img_filepath, 1), (192,192))
      img_list_n.append(img)
      label_img = np.zeros((img_size[0], img_size[1]), np.uint8)
      label_list_n.append(label_img)

    else:
      img = cv2.resize(cv2.imread(img_filepath, 1), (192,192))
      img_list_l.append(img)
      f = open(json_filepath, 'r', encoding='utf-8')
      json_data = json.load(f)
      json_img_size = (json_data['imageHeight'], json_data['imageWidth'])
      label_img = np.zeros((img_size[0], img_size[1]), np.uint8)
      for json_shape in json_data['shapes']:
        label = json_shape['label']
        if not label in flag_list: flag_list.append(label)
        points = json_shape['points']
        for point in points:
          point[0] = point[0] * (img_size[1]/json_img_size[1])
          point[1] = point[1] * (img_size[0]/json_img_size[0])
          if point[0] < 0: point[0] = 0
          if point[0] > img_size[1]: point[0] = img_size[1]
          if point[1] < 0: point[1] = 0
          if point[1] > img_size[0]: point[1] = img_size[0]
        if json_shape['shape_type'] == 'polygon':
          shape_contour = np.array([points], np.int32)
          flag_index = flag_list.index(label) + 1
          label_img = cv2.fillPoly(label_img, pts=shape_contour, color=1) #color=label_index)
      f.close()
      label_list_l.append(label_img)

  train_end_index_l = len(img_list_l) - int(len(img_list_l) * valid_split)
  train_end_index_n = len(img_list_l) - int(len(img_list_n) * valid_split)

  train_img_list = img_list_l[:train_end_index_l] + img_list_n[:train_end_index_n]
  train_label_list = label_list_l[:train_end_index_l] + label_list_n[:train_end_index_n]
  valid_img_list = img_list_l[train_end_index_l:] + img_list_n[train_end_index_n:]
  valid_label_list = label_list_l[train_end_index_l:] + label_list_n[train_end_index_n:]

  train_img_arr = np.array(train_img_list).astype(np.float32)/ 255.
  train_label_arr = np.array(train_label_list).astype(np.float32)/ 1.
  valid_img_arr = np.array(valid_img_list).astype(np.float32)/ 255.
  valid_label_arr = np.array(valid_label_list).astype(np.float32)/ 1.
  
  return train_img_arr, train_label_arr, valid_img_arr, valid_label_arr

def monitor_dataset(images, labels):
  for (img,lbl) in zip(images, labels):
    cv2.imshow("img",img) 
    lbl = lbl * 255
    cv2.imshow("lbl",lbl) 
    key = cv2.waitKey(0)
    if key == 27:
      break
  cv2.destroyAllWindows()
    