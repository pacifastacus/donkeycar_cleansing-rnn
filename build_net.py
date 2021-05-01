import keras 
from keras.models import load_model, save_model
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Concatenate, LSTM
from keras.layers.wrappers import TimeDistributed as TD
from matplotlib.image import imread
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import json

TEST_DATA_DIR = data_dir = "/media/users/DATA/Projektek/classify-donkey-dataset/test_data/img"

def load_image_subnet(file='rnn_cnn_top.h5'):
	model = load_model(file, compile=False)
	model.trainable = False
	return model
	
def get_img_jsonlist_ordered(data_dir):
	json_list = glob.glob(data_dir + '/record_[0-9]*.json')
	json_list.sort(key= lambda fname: int(os.path.basename(fname).split('record_')[1].split('.json')[0]))
	return json_list

def make_control_signal_subnet(seq_len=9):
	control_seq_shape = (9,2)
	x = Sequential()
	x.add(TD(Input(shape=control_seq_shape)))
	return x

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

		return imread(os.path.join(data_dir,im_name))

def build_model():
	img_net = load_image_subnet()
	img_net_in = img_net.input
	img_net_out = img_net.output

	ctrl_in = Input(shape=(9,2),name="ctrl_in")

	x = Concatenate(axis=2)([ctrl_in,img_net_out])
	x = LSTM(128, return_sequences=True, name="LSTM_seq")(x)
	x = Dropout(0.1)(x)
	x = LSTM(128, return_sequences=False, name="LSTM_out")(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(64, activation='relu')(x)
	x = Dense(10, activation='relu')(x)
	out = Dense(1, activation='linear', name='model_outputs')(x)
	model = Model(inputs=[img_net_in,ctrl_in],outputs=out, name="LSTM_dataset_classifier")
	return model

def make_plot(model):
	from keras.utils import plot_model
	plot_model(model,model.name+".png",show_shapes=True,show_dtype=True,show_layer_names=True,expand_nested=True,dpi=300)

