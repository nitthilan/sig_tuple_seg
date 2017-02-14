import cv2
import numpy as np
import time
import os

def get_train_list(basepath):
	train_file_list = os.listdir(basepath)
	train_img_list = []
	train_mask_list = []
	for file in train_file_list:
		if(file[-8:] == "mask.jpg"):
			train_mask_list.append(file);
		else:
			train_img_list.append(file)
	return (train_img_list, train_mask_list)

def get_numpy_array(img_filepath):
	X_train = np.empty((num_images, color_mode, width, height))
	im = Image.open(img_filepath)
	imarray = np.array(im)
	return imarray

def get_im_cv2(path, color_type=3):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    return img


def get_train_images_old(basepath):
	train_img_list, train_mask_list = get_train_list(basepath)
	num_images = len(train_img_list)
	width = height = 128
	train_img = np.empty((num_images, 3, width, height))
	train_mask = np.empty((num_images, 1, width, height))
	num_image = 0
	for file in train_img_list:
		img = get_im_cv2(basepath+file)
		train_img[num_image][0] = img[:,:,0]
		train_img[num_image][1] = img[:,:,1]
		train_img[num_image][2] = img[:,:,2]
		num_image += 1

	num_image = 0
	for file in train_mask_list:
		img = get_im_cv2(basepath+file, 1)
		ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		train_mask[num_image][0] = img[:,:]
		num_image += 1

	return [train_img, train_mask, train_img_list]

def get_train_images(basepath, filter_size, step_size):
	train_img_list, train_mask_list = get_train_list(basepath)
	num_images = len(train_img_list)
	train_img_array = []
	train_img_mask_array = []
	num_image = 0
	for file in train_img_list:
		mask_file_name = file[:-4]+"-mask.jpg"
		img = get_im_cv2(basepath+file)
		img_mask = get_im_cv2(basepath+mask_file_name, 1)
		ret,img_mask = cv2.threshold(img_mask,127,1,cv2.THRESH_BINARY)

		(width, height, num_color) = img.shape;
		
		if(width%filter_size):
			padded_width = int((width+filter_size)/filter_size)*filter_size
		else:
			padded_width = width;

		if(height%filter_size):
			padded_height = int((height+filter_size)/filter_size)*filter_size
		else:
			padded_height = height;
		#padded_width = padded_height = 128
		padded_img = np.zeros((3, padded_width, padded_height))
		for i in [0,1,2]:
			padded_img[i,0:width,0:height] = img[:,:, i]
		#print("padded values ", padded_width, padded_height, width, height, num_image, padded_img.shape)

		padded_img_mask = np.zeros((1, padded_width, padded_height))
		padded_img_mask[0, 0:width, 0:height] = img_mask

		for j in range(0, padded_width-filter_size+1, step_size):
			for k in range(0, padded_height-filter_size+1, step_size):				
				train_img_array.append(padded_img[:, j:j+filter_size, k:k+filter_size])
				train_img_mask_array.append(padded_img_mask[:, j:j+filter_size, k:k+filter_size])
				#print("Dime ", train_img_array[-1].shape, j, k)
				num_image+=1
	train_img_array = np.array(train_img_array)
	train_img_mask_array = np.array(train_img_mask_array)
	#print("Size ", )
	return (train_img_array, train_img_mask_array, train_img_list)





def get_train_images_dwn_smpl(basepath, downscale_factor, dimension):
	train_img_list, train_mask_list = get_train_list(basepath)
	num_images = len(train_img_list)
	width = height = dimension
	total_images = downscale_factor*downscale_factor*num_images
	resize_width = width/downscale_factor;
	resize_height = height/downscale_factor;

	train_img = np.empty((total_images, 3, resize_width, resize_height))
	num_image = 0
	for file in train_img_list:
		img = get_im_cv2(basepath+file)
		for j in range(downscale_factor):
			for k in range(downscale_factor):
				for i in [0,1,2]:
					train_img[num_image][i] = \
						img[j*resize_width:(j+1)*resize_width, k*resize_height:(k+1)*resize_height, i]
				num_image += 1

	train_mask = np.empty((total_images, 1, resize_width, resize_height))
	num_image = 0
	for file in train_mask_list:
		img = get_im_cv2(basepath+file, 1)
		ret,img = cv2.threshold(img,127,1,cv2.THRESH_BINARY)
		# print (img[64:64+16, 64:64+16], img[0:16, 0:16])
		for j in range(downscale_factor):
			for k in range(downscale_factor):
				train_mask[num_image][0] = \
					img[j*resize_width:(j+1)*resize_width, k*resize_height:(k+1)*resize_height]
				num_image += 1

	return [train_img, train_mask, train_img_list]

def get_test_images_dwn_scled(basepath, downscale_dimension):
	test_file_list = os.listdir(basepath)
	#test_file_list = test_file_list[:1]

	test_list = []
	test_file_dimension = []
	num_image = 0
	for file in test_file_list:
		img = get_im_cv2(basepath+file)
		(width, height, num_color) = img.shape;
		test_file_dimension.append((width, height, num_color))
		padded_width = int((width+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_height = int((height+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_img = np.zeros((3, padded_width, padded_height))
		for i in [0,1,2]:
			padded_img[i,0:width,0:height] = img[:,:, i]
		print(width, height, padded_width, padded_height, file)
		for j in range(0, padded_height, downscale_dimension):
			for k in range(0, padded_width, downscale_dimension):				
				test_list.append(padded_img[:, k:k+downscale_dimension, j:j+downscale_dimension])
		num_image+=1
	return [np.array(test_list), test_file_list, test_file_dimension]

def get_test_images_ovr_lap(basepath, downscale_dimension):
	test_file_list = os.listdir(basepath)
	#test_file_list = test_file_list[:3]

	test_list = []
	test_file_dimension = []
	num_image = 0
	step_size = downscale_dimension/2
	for file in test_file_list:
		if(file[-4:] != ".jpg"):
			continue;
		img = get_im_cv2(basepath+file)
		print (file)
		(width, height, num_color) = img.shape;
		test_file_dimension.append((width, height, num_color))
		padded_width = int((width+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_height = int((height+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_img = np.zeros((3, padded_width, padded_height))
		for i in [0,1,2]:
			padded_img[i,0:width,0:height] = img[:,:, i]
		print(width, height, padded_width, padded_height, file)
		for j in range(0, padded_height-downscale_dimension+1, step_size):
			for k in range(0, padded_width-downscale_dimension+1, step_size):				
				test_list.append(padded_img[:, k:k+downscale_dimension, j:j+downscale_dimension])
		num_image+=1
	return [np.array(test_list), test_file_list, test_file_dimension]

def get_test_images_ovr_lap_off(basepath, downscale_dimension):
	test_file_list = os.listdir(basepath)
	test_file_list = test_file_list[:1]

	test_list = []
	test_file_dimension = []
	num_image = 0
	step_size = downscale_dimension/8
	for file in test_file_list:
		img = get_im_cv2(basepath+file)
		(width, height, num_color) = img.shape;
		test_file_dimension.append((width, height, num_color))
		padded_width = int((width+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_height = int((height+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_img = np.zeros((3, padded_width, padded_height))
		for i in [0,1,2]:
			padded_img[i,0:width,0:height] = img[:,:, i]
		print(width, height, padded_width, padded_height, file)
		for j in range(0, padded_height-downscale_dimension+1, step_size):
			for k in range(0, padded_width-downscale_dimension+1, step_size):				
				test_list.append(padded_img[:, k:k+downscale_dimension, j:j+downscale_dimension])
		num_image+=1
	return [np.array(test_list), test_file_list, test_file_dimension]


def dump_test_images_dwn_scled(basepath, test_file_list, test_file_dimension, input_image_list, 
	downscale_dimension):#, input_basepath):
	#test_file_list = os.listdir(basepath)
	test_list = []
	num_image = 0
	num_image_id = 0
	if(not os.path.isdir(basepath)):
		print(" Did not find directory ", basepath)
		os.makedirs(os.path.isdir)
	for file in test_file_list:	
		(width, height, num_color) = test_file_dimension[num_image]
	
		padded_width = int((width+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_height = int((height+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_img = np.zeros((padded_width, padded_height))

		print(width, height, padded_width, padded_height, file)
		for j in range(0, padded_height, downscale_dimension):
			for k in range(0, padded_width, downscale_dimension):
				padded_img[k:k+downscale_dimension, j:j+downscale_dimension] = \
						input_image_list[num_image_id];
				num_image_id+=1

		img = padded_img[0:width, 0:height]
		img[img<0.5] = 0;
		img[img>=0.5] = 255;
				
		print("Mask File ", img.shape)
		cv2.imwrite(basepath+file[:-4]+"-mask.jpg", img);
		# img_1 = get_im_cv2(input_basepath+file)
		# print("Validate ", img.shape, img_1.shape)
		# print("Compare output", file, np.array_equal(img[0,:,:], img_1[:,:,0]))
		num_image+=1
	return

def dump_test_images_ovr_lpd(basepath, test_file_list, test_file_dimension, input_image_list, 
	downscale_dimension):#, input_basepath):
	#test_file_list = os.listdir(basepath)
	test_list = []
	num_image = 0
	num_image_id = 0
	step_size = downscale_dimension/2
	if(not os.path.isdir(basepath)):
		print(" Did not find directory ", basepath)
		os.makedirs(os.path.isdir)
	for file in test_file_list:	
		if(file[-4:] != ".jpg"):
			continue;
		(width, height, num_color) = test_file_dimension[num_image]
	
		padded_width = int((width+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_height = int((height+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_img = np.zeros((padded_width, padded_height))

		print(width, height, padded_width, padded_height, file)
		for j in range(0, padded_height-downscale_dimension+1, step_size):
			for k in range(0, padded_width-downscale_dimension+1, step_size):
				start = step_size/2; end = downscale_dimension-(step_size/2)
				#print("Shape ", input_image_list.shape, start, end)
				padded_img[k+start:k+end, j+start:j+end] = \
						input_image_list[num_image_id, 0, start:end, start:end];
				num_image_id+=1

		img = padded_img[0:width, 0:height]
		# img[img<0.5] = 0;
		# img[img>=0.5] = 255;

		# img *= 255
		# img[img<0] = 0
		# img[img>=255] = 255

		img[img<0.5] = 0;
		img[img>=0.5] = 255;

		print("Mask File ", img.shape)
		cv2.imwrite(basepath+file[:-4]+"-mask.jpg", img);
		# img_1 = get_im_cv2(input_basepath+file)
		# print("Validate ", img.shape, img_1.shape)
		# print("Compare output", file, np.array_equal(img[0,:,:], img_1[:,:,0]))
		num_image+=1
	return

def dump_test_images_ovr_lpd_off(basepath, test_file_list, test_file_dimension, input_image_list, 
	downscale_dimension):#, input_basepath):
	#test_file_list = os.listdir(basepath)
	test_file_list = test_file_list[:1]
	test_list = []
	num_image = 0
	num_image_id = 0
	step_size = downscale_dimension/8
	scale_filter = np.zeros((downscale_dimension, downscale_dimension))
	scale_filter[16:48, 16:48] = 0.2
	scale_filter[24:40, 24:40] = 0.4
	
	if(not os.path.isdir(basepath)):
		print(" Did not find directory ", basepath)
		os.makedirs(os.path.isdir)
	for file in test_file_list:	
		(width, height, num_color) = test_file_dimension[num_image]
	
		padded_width = int((width+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_height = int((height+downscale_dimension)/downscale_dimension)*downscale_dimension
		padded_img = np.zeros((padded_width, padded_height))

		print(width, height, padded_width, padded_height, file)
		for j in range(0, padded_height-downscale_dimension+1, step_size):
			for k in range(0, padded_width-downscale_dimension+1, step_size):
				start = step_size/2; end = downscale_dimension-(step_size/2)
				#print("Shape ", input_image_list.shape, start, end)
				padded_img[k+start:k+end, j+start:j+end] = \
						(input_image_list[num_image_id, 0, start:end, start:end] * \
						scale_filter);
				num_image_id+=1

		img = padded_img[0:width, 0:height]
		# img[img<0.5] = 0;
		# img[img>=0.5] = 255;

		# img *= 255
		# img[img<0] = 0
		# img[img>=255] = 255

		img[img<0.0] = 0;
		img[img>=0.0] = 255;

		print("Mask File ", img.shape)
		cv2.imwrite(basepath+file[:-4]+"-mask.jpg", img);
		# img_1 = get_im_cv2(input_basepath+file)
		# print("Validate ", img.shape, img_1.shape)
		# print("Compare output", file, np.array_equal(img[0,:,:], img_1[:,:,0]))
		num_image+=1
	return



def get_test_images(basepath, color_type=3):
	test_file_list = os.listdir(basepath)
	test_list = []
	final_file_list = []
	for file in test_file_list:
		if not file.startswith('.'):
			test_list.append(get_im_cv2(basepath+file, color_type))
			final_file_list.append(file)
	return (test_list, final_file_list)


def get_cached_data(cachepath, enablecache):
    if(os.path.isfile(cachepath) and enablecache):
        loaded = np.load(cachepath)
        #print loaded.files
        return (loaded['arr_0'], True)
        # loaded = h5py.File(cachepath+".h5", "r")
        # return (loaded['train_80_60_3_22424'][()], True)
    print('Cache Data Not Found. Loading Data from source ....')
    return (None, False)

def cache_data(cachepath, enablecache, x):
    if(os.path.isdir(os.path.dirname(cachepath)) and enablecache):
        with open(cachepath, "wb") as cachefile:
            np.savez(cachefile, x)
    else:
        print('Data Not cached. directory not present or cache disabled')

def get_image_cached(basepath, get_img_func, cachepath, enablecache):
	start_time = time.time()
	print cachepath
	(x, iscached) = get_cached_data(cachepath, enablecache)
	if(not iscached):
	    x = get_img_func(basepath)
	    cache_data(cachepath, enablecache, x)
	print "Time to read test data", time.time() - start_time

	return x

def get_images(basepath, get_img_func):
	start_time = time.time()
	x = get_img_func(basepath)	
	print "Time to read test data", time.time() - start_time
	return x



DATA_BASE_PATH = "../../../data/sig_tuple_seg/SigTuple_data/"
DATA_CACHE_BASE_PATH = "../../../data/sig_tuple_seg/cache/"

TRAIN_PATH = DATA_BASE_PATH+"Train_Data/"
TEST_PATH = DATA_BASE_PATH+"Test_Data/"
TEST_CACHE_PATH = DATA_CACHE_BASE_PATH+"test.h5"
TRAIN_CACHE_PATH = DATA_CACHE_BASE_PATH+"train.h5"


if __name__ == "__main__":
	x = get_images(TRAIN_PATH, get_train_images)
	x = get_images(TEST_PATH, get_test_images)


