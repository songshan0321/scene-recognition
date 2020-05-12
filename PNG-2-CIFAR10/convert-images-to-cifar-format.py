#python script for converting 32x32 pngs to format
from PIL import Image
import os
from array import *
import pickle
import numpy as np

annotation_dict = {}
labels = []
data = []

for dirname, dirnames, filenames in os.walk('./classes'):
	for filename in filenames:
		if filename.endswith('.png'):

			################
			#grab the image#
			################

			im = np.asarray(Image.open(os.path.join(dirname, filename))).transpose(2,0,1)
			# pix = im.load()
			#print(os.path.join(dirname, filename))

			#store the class name from look at path
			class_name = int(os.path.join(dirname).split('/')[-1])
			#print class_name

			###########################
			#get image into byte array#
			###########################

			# create array of bytes to hold stuff

			#first append the class_name byte
			labels.append(class_name)
			data.append(im)

			#then write the rows
			#Extract RGB from pixels and append
			#note: first we get red channel, then green then blue
			#note: no delimeters, just append for all images in the set
			# for color in range(0,3):
			# 	for x in range(0,256):
			# 		for y in range(0,256):
			# 			data.append(pix[x,y][color])


############################################
#write all to binary, all set for cifar10!!#
############################################
labels = np.array(labels)
data = np.array(data)

print(labels)
print(labels.shape)
print(data.shape)

annotation_dict['labels'] = labels
annotation_dict['data'] = data
output_file = open('val', 'wb')
pickle.dump(annotation_dict, output_file)
output_file.close()