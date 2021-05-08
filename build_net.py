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

TEST_DATA_DIR = data_dir = "/media/users/DATA/Projektek/donkeycar-data_cleansing/test_data/img"

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
	
# def seq_generator(data_dir, json_list, seq_len=9):
# 	img_seq = []
# 	ctrl_seq = []
# 	# Preload img_seq
# 	with open(json_list[0]) as f:
# 		datas = json.load(f)
# 	while len(img_seq) < seq_len:
# 		im = get_img(data_dir,datas["cam/image_array"])
# 		ctrl = datas["user/throttle"], datas["user/angle"]
# 		img_seq.append(im)
# 		ctrl_seq.append(ctrl)
# 	# Yield the next seq_len-th image
# 	for i in range(len(json_list)):
# 		with open(json_list[i]) as f:
# 			#im = get_img(json_list[i])
# 			datas = json.load(f)
# 			im = get_img(data_dir,datas["cam/image_array"])
# 			ctrl = datas["user/throttle"], datas["user/angle"]
# 			img_seq = img_seq[1:]
# 			img_seq.append(im)
# 			ctrl_seq = ctrl_seq[1:]
# 			ctrl_seq.append(ctrl)
# 			yield np.array(img_seq).reshape(1,seq_len,*im.shape), np.array(ctrl_seq).reshape(1,seq_len,2)

def data_seq_from_json_list(json_list,data_dir):
	img_seq = []
	ctrl_seq = []
	path,_ = os.path.split(data_dir)
	for filename in json_list:
		with open(os.path.join(path,filename)) as f:
			df = json.load(f)
		im   = get_img(data_dir,df["cam/image_array"])
		ctrl = df["user/throttle"],df["user/angle"]
		img_seq.append(im)
		ctrl_seq.append(ctrl)
	return np.array(img_seq), np.array(ctrl_seq)

def data_generator(data_dir, seq_len=9):
	path, basename = os.path.split(data_dir)
	table = {
		"json":[],
		"y":[]
	}
	with open(os.path.join(path,basename+"_filter.csv")) as f:
		f.readline()	#Skip the header
		for line in f:
			json_name, y = line.strip().split(",")
			table["json"].append(json_name)
			table["y"].append(int(y))

	for i in range(seq_len-1,len(table["y"])):
		X = data_seq_from_json_list(table["json"][i-seq_len+1:i+1],data_dir)
		yield X,table["y"][i]


def get_img(data_dir,im_name):
#	with open(im) as f:
#		im_name = json.load(f)["cam/image_array"]
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

class DataGen(keras.utils.Sequence):
	def __init__(self, df,X_col,y_col,batch_size):
		self.batch_size = batch_size
		self.df = df
		self.X_col = X_col
		self.y_col = y_col
		pass

	def __get_data(self,batches):

		return X,y

	def __getitem__(self, item):
		'''
		Assemble and load the next mini-batch
		:param item: the item-th minibatch from the whole dataset
		:return: a mini-batch
		'''

		batches = self.df[item*self.batch_size:(item+1)*self.batch_size]
		X, y = self.__get_data(batches)
		return  X, y

	def on_epoch_end(self):
		pass

	def __len__(self):
		pass

