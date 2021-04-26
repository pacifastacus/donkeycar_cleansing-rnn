from matplotlib import pyplot as plt
import numpy as np

from build_net import *

plt.ion()

model = load_image_subnet()
files_list = get_img_flist_ordered(TEST_DATA_DIR)

x_bar = [x for x in range(128)]

for seq in seq_generator(files_list):
	output = model.predict(seq)
	plt.subplot(2,1,1)
	plt.imshow(np.concatenate((seq[0,0],seq[0,4],seq[0,8]),axis=1))
	plt.subplot(2,1,2)
	plt.bar(x_bar,output.flatten())
	axes = plt.gca()
	axes.set_ybound(-1,1)
	axes.set_xbound(0,127)
	plt.pause(0.0625)
	plt.clf()
