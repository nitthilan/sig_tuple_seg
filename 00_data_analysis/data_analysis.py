from os import listdir
from PIL import Image
import numpy as np
import cv2

DATA_BASE_PATH = "../../../data/sig_tuple_seg/SigTuple_data_new/"

TRAIN_PATH = DATA_BASE_PATH+"Train_Data/"
TEST_PATH = DATA_BASE_PATH+"Test_Data/"


train_file_list = listdir(TRAIN_PATH)
train_img_list = []
train_mask_list = []
for file in train_file_list:
	if(file[-8:] == "mask.jpg"):
		train_mask_list.append(file);
	else:
		train_img_list.append(file)

# find the area of each contour of the segmented images

for file in train_mask_list:
	file_path = TRAIN_PATH+file
	img = cv2.imread(file_path)
	img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(img1,127,255,0)
	(contours, hierarchy) = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print(file)
	for cnt in contours:
		M = cv2.moments(cnt)
		area = cv2.contourArea(cnt)
		perimeter = cv2.arcLength(cnt,True)
		if(area < 500):
			print("Area ", area, 
				int(M['m10']/M['m00']), int(M['m01']/M['m00']))



# im = Image.open(TRAIN_PATH+train_img_list[0])
# im.show()
# mask_file = TRAIN_PATH+train_mask_list[0]
# print mask_file
# im_mask = Image.open(mask_file)
# im_mask.show()
# test_file_list = listdir(TEST_PATH)
# for file in test_file_list:
# 	im = Image.open(TEST_PATH+file)
# 	imarray = np.array(im)
# 	print file, imarray.shape, imarray.size#, imarray

# print len(test_file_list)

