import keras 
from keras.models import load_model, save_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from matplotlib.image import imread
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import json

TEST_DATA_DIR = data_dir = "/media/users/DATA/Projektek/classify-donkey-dataset/test_data/img"

def load_image_subnet(file='rnn_top.h5'):
	model = load_model(file,compile=False)
	model.trainable = False
	return model
	
def make_control_signal_subnet():
	pass


def get_img_jsonlist_ordered(data_dir):
	json_list = glob.glob(data_dir + '/record_[0-9]*.json')
	json_list.sort(key= lambda fname: int(os.path.basename(fname).split('record_')[1].split('.json')[0]))
	return json_list

def get_img_flist_ordered(data_dir):
	im_list = glob.glob(data_dir+'/*.jpg')
	im_list.sort(key= lambda fname: int(os.path.basename(fname).split('_')[0]))
	return im_list
	
def seq_generator(img_list,seq_len=9):
	img_seq = []
	# Preload img_seq
	while len(img_seq) < seq_len:
		im = get_img(img_list[0])
		img_seq.append(im)
	
	# Yield the next seq_len-th image
	for i in range(len(img_list)):
		im = get_img(img_list[i])
		img_seq = img_seq[1:]
		img_seq.append(im)
		yield np.array(img_seq).reshape(1,seq_len,*im.shape)
	
	
def get_img(im):
	with open(im) as f:
		im_name = json.load(f)["cam/image_array"]
		return imread(data_dir + im_name)