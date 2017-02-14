from os import listdir
from PIL import Image
import numpy as np
import data_loading as dl
import models as mo
from keras.callbacks import ModelCheckpoint
import os
#from sklearn.cross_validation import KFold


DATA_BASE_PATH = "../../../data/sig_tuple_seg/SigTuple_data_new/"
DATA_CACHE_BASE_PATH = "../../../data/sig_tuple_seg/cache/"

TRAIN_PATH = DATA_BASE_PATH+"Train_Data/"
TEST_PATH = DATA_BASE_PATH+"Test_Data/"
TEST_CACHE_PATH = DATA_CACHE_BASE_PATH+"test.h5"
TRAIN_CACHE_PATH = DATA_CACHE_BASE_PATH+"train.h5"
WEIGHT_CACHE_PATH = DATA_CACHE_BASE_PATH+"weight_cache.h5"
WEIGHT_OLD_CACHE_PATH = DATA_CACHE_BASE_PATH+"weight_cache_64_1_20.h5" #na is not available


n_folds = 10;
random_state = 51
dimension = 128
num_color_component = 3
filter_size = 64
step_size = 16
train_factor = 0.9
batch_size = 128
nb_epoch = 30



# scaled_dimension = dimension/downscale_factor

#[train_img, train_mask, train_img_list] = dl.get_images(TRAIN_PATH, dl.get_train_images)
# [train_img, train_mask, train_img_list] = \
# 	dl.get_train_images_dwn_smpl(TRAIN_PATH, downscale_factor, dimension)

(train_img, train_mask, train_img_list) = \
	dl.get_train_images(TRAIN_PATH, filter_size, step_size)

print("Mean R, G, B, M", np.mean(train_img[:,0,:,:]), np.mean(train_img[:,1,:,:]), 
	np.mean(train_img[:,2,:,:]), np.mean(train_mask[:,0,:,:]) )

# print("Mean R, G, B, M", np.mean(train_img_1[:,0,:,:]), np.mean(train_img_1[:,1,:,:]), 
# 	np.mean(train_img_1[:,2,:,:]), np.mean(train_mask_1[:,0,:,:]) )

# train_img[:,0,:,:] -= 187
# train_img[:,1,:,:] -= 171
# train_img[:,2,:,:] -= 182
# train_img = train_img/128;

# train_img = train_img - 128;
# train_img = train_img/128;

print(train_img.shape, train_mask.shape, len(train_img_list), train_img.shape[0])

model = mo.get_unet(num_color_component, filter_size);

if os.path.isfile(WEIGHT_OLD_CACHE_PATH):
    model.load_weights(WEIGHT_OLD_CACHE_PATH)
else:
	print("No model loaded!")

num_images = train_img.shape[0]
num_train_images = int(num_images*train_factor)
num_valid_images = num_images - num_train_images
print("Num images total, train, valid ", num_images, num_train_images, num_valid_images)
train_x = train_img[0:num_train_images]
valid_x = train_img[num_train_images:num_images]
train_y = train_mask[0:num_train_images]
valid_y = train_mask[num_train_images:num_images]

print("Mean R, G, B", np.mean(train_x[:,0,:,:]), np.mean(train_x[:,1,:,:]), np.mean(train_x[:,2,:,:]) )


# print(train_x.shape, valid_x.shape)
# print(train_x[36], train_y[36], train_img_list[0])
# print(valid_x[60], valid_y[60], train_img_list[147])


weights_path = WEIGHT_CACHE_PATH
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    #TensorBoard(os.path.join(infobasepath,'logs'), histogram_freq=1, write_graph=True)
]


model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch,
      shuffle=True, verbose=1, validation_data=(valid_x, valid_y),
      callbacks=callbacks)


# kf = KFold(len(train_img_list), n_folds=n_folds, shuffle=True, random_state=random_state)
# print (kf)
# for train, valid in kf:
# 	print (len(train), len(valid))

# Unet filter size of 64. Training vs Validation result
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ('Mean R, G, B, M', 183.61820537735662, 177.39776022324151, 183.90411019355651, 0.09403537428634777)
# ((18041, 3, 64, 64), (18041, 1, 64, 64), 169, 18041)
# ('Num images total, train, valid ', 18041, 16236, 1805)
# ('Mean R, G, B', 183.23607705232104, 180.03622619222907, 185.38108393745958)
# Train on 16236 samples, validate on 1805 samples
# Epoch 1/30
# 16236/16236 [==============================] - 3229s - loss: -0.4525 - dice_coef: 0.4525 - val_loss: -0.8755 - val_dice_coef: 0.8755
# Epoch 2/30
# 16236/16236 [==============================] - 3623s - loss: -0.8498 - dice_coef: 0.8498 - val_loss: -0.8999 - val_dice_coef: 0.8999
# Epoch 3/30
# 16236/16236 [==============================] - 3723s - loss: -0.8818 - dice_coef: 0.8818 - val_loss: -0.9118 - val_dice_coef: 0.9118
# Epoch 4/30
# 16236/16236 [==============================] - 3631s - loss: -0.9050 - dice_coef: 0.9050 - val_loss: -0.9172 - val_dice_coef: 0.9172
# Epoch 5/30
# 16236/16236 [==============================] - 4332s - loss: -0.9164 - dice_coef: 0.9164 - val_loss: -0.9214 - val_dice_coef: 0.9214
# Epoch 6/30
# 16236/16236 [==============================] - 3184s - loss: -0.9225 - dice_coef: 0.9225 - val_loss: -0.9235 - val_dice_coef: 0.9235
# Epoch 7/30
# 16236/16236 [==============================] - 2911s - loss: -0.9272 - dice_coef: 0.9272 - val_loss: -0.9279 - val_dice_coef: 0.9279
# Epoch 8/30
# 16236/16236 [==============================] - 13522s - loss: -0.9325 - dice_coef: 0.9325 - val_loss: -0.9318 - val_dice_coef: 0.9318
# Epoch 9/30
# 16236/16236 [==============================] - 4167s - loss: -0.9322 - dice_coef: 0.9322 - val_loss: -0.9306 - val_dice_coef: 0.9306
# Epoch 10/30
# 16236/16236 [==============================] - 5038s - loss: -0.9368 - dice_coef: 0.9368 - val_loss: -0.9314 - val_dice_coef: 0.9314
# Epoch 11/30
# 16236/16236 [==============================] - 4495s - loss: -0.9392 - dice_coef: 0.9392 - val_loss: -0.9327 - val_dice_coef: 0.9327
# Epoch 12/30
# 16236/16236 [==============================] - 3828s - loss: -0.9404 - dice_coef: 0.9404 - val_loss: -0.9349 - val_dice_coef: 0.9349
# Epoch 13/30
# 16236/16236 [==============================] - 4705s - loss: -0.9432 - dice_coef: 0.9432 - val_loss: -0.9389 - val_dice_coef: 0.9389
# Epoch 14/30
# 16236/16236 [==============================] - 3485s - loss: -0.9451 - dice_coef: 0.9451 - val_loss: -0.9388 - val_dice_coef: 0.9388
# Epoch 15/30
# 16236/16236 [==============================] - 3793s - loss: -0.9459 - dice_coef: 0.9459 - val_loss: -0.9359 - val_dice_coef: 0.9359
# Epoch 16/30
# 16236/16236 [==============================] - 5891s - loss: -0.9485 - dice_coef: 0.9485 - val_loss: -0.9351 - val_dice_coef: 0.9351
# Epoch 17/30
# 16236/16236 [==============================] - 6547s - loss: -0.9481 - dice_coef: 0.9481 - val_loss: -0.9367 - val_dice_coef: 0.9367
# Epoch 18/30






# Output for filter size 16
# Train on 9446 samples, validate on 1050 samples
# Epoch 1/10
# 9446/9446 [==============================] - 279s - loss: -0.3579 - dice_coef: 0.3579 - val_loss: -0.5076 - val_dice_coef: 0.5076
# Epoch 2/10
# 9446/9446 [==============================] - 237s - loss: -0.6203 - dice_coef: 0.6203 - val_loss: -0.6682 - val_dice_coef: 0.6682
# Epoch 3/10
# 9446/9446 [==============================] - 228s - loss: -0.7112 - dice_coef: 0.7112 - val_loss: -0.7299 - val_dice_coef: 0.7299
# Epoch 4/10
# 9446/9446 [==============================] - 227s - loss: -0.8000 - dice_coef: 0.8000 - val_loss: -0.8242 - val_dice_coef: 0.8242
# Epoch 5/10
# 9446/9446 [==============================] - 271s - loss: -0.8305 - dice_coef: 0.8305 - val_loss: -0.8460 - val_dice_coef: 0.8460
# Epoch 6/10
# 9446/9446 [==============================] - 252s - loss: -0.8365 - dice_coef: 0.8365 - val_loss: -0.8532 - val_dice_coef: 0.8532
# Epoch 7/10
# 9446/9446 [==============================] - 224s - loss: -0.8385 - dice_coef: 0.8385 - val_loss: -0.8527 - val_dice_coef: 0.8527
# Epoch 8/10
# 9446/9446 [==============================] - 241s - loss: -0.8412 - dice_coef: 0.8412 - val_loss: -0.8541 - val_dice_coef: 0.8541
# Epoch 9/10
# 9446/9446 [==============================] - 277s - loss: -0.8437 - dice_coef: 0.8437 - val_loss: -0.8521 - val_dice_coef: 0.8521
# Epoch 10/10
# 9446/9446 [==============================] - 274s - loss: -0.8459 - dice_coef: 0.8459 - val_loss: -0.8547 - val_dice_coef: 0.8547

# Output for filter size 32
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ((2624, 3, 32, 32), (2624, 1, 32, 32), 164, 2624)
# (61, 61)
# ('Num images total, train, valid ', 2624, 2361, 263)
# Train on 2361 samples, validate on 263 samples
# Epoch 1/10
# 2361/2361 [==============================] - 171s - loss: -0.2752 - dice_coef: 0.2752 - val_loss: -0.3446 - val_dice_coef: 0.3446
# Epoch 2/10
# 2361/2361 [==============================] - 166s - loss: -0.4425 - dice_coef: 0.4425 - val_loss: -0.5207 - val_dice_coef: 0.5207
# Epoch 3/10
# 2361/2361 [==============================] - 156s - loss: -0.5499 - dice_coef: 0.5499 - val_loss: -0.5849 - val_dice_coef: 0.5849
# Epoch 4/10
# 2361/2361 [==============================] - 172s - loss: -0.6010 - dice_coef: 0.6010 - val_loss: -0.6326 - val_dice_coef: 0.6326
# Epoch 5/10
# 2361/2361 [==============================] - 157s - loss: -0.6587 - dice_coef: 0.6587 - val_loss: -0.7041 - val_dice_coef: 0.7041
# Epoch 6/10
# 2361/2361 [==============================] - 150s - loss: -0.7302 - dice_coef: 0.7302 - val_loss: -0.7695 - val_dice_coef: 0.7695
# Epoch 7/10
# 2361/2361 [==============================] - 174s - loss: -0.7815 - dice_coef: 0.7815 - val_loss: -0.8051 - val_dice_coef: 0.8051
# Epoch 8/10
# 2361/2361 [==============================] - 153s - loss: -0.8129 - dice_coef: 0.8129 - val_loss: -0.8270 - val_dice_coef: 0.8270
# Epoch 9/10
# 2361/2361 [==============================] - 169s - loss: -0.8344 - dice_coef: 0.8344 - val_loss: -0.8348 - val_dice_coef: 0.8348
# Epoch 10/10
# 2361/2361 [==============================] - 138s - loss: -0.8485 - dice_coef: 0.8485 - val_loss: -0.8374 - val_dice_coef: 0.8374


#Output for filter size 64
# Using Theano backend.
# ((656, 3, 64, 64), (656, 1, 64, 64), 164, 656)
# (61, 61)
# ('Num images total, train, valid ', 656, 590, 66)
# Train on 590 samples, validate on 66 samples
# Epoch 1/10
# 590/590 [==============================] - 133s - loss: -0.3130 - dice_coef: 0.3130 - val_loss: -0.3221 - val_dice_coef: 0.3221
# Epoch 2/10
# 590/590 [==============================] - 129s - loss: -0.3147 - dice_coef: 0.3147 - val_loss: -0.3234 - val_dice_coef: 0.3234
# Epoch 3/10
# 590/590 [==============================] - 125s - loss: -0.3158 - dice_coef: 0.3158 - val_loss: -0.3241 - val_dice_coef: 0.3241
# Epoch 4/10
# 590/590 [==============================] - 165s - loss: -0.3164 - dice_coef: 0.3164 - val_loss: -0.3245 - val_dice_coef: 0.3245
# Epoch 5/10
# 590/590 [==============================] - 141s - loss: -0.3168 - dice_coef: 0.3168 - val_loss: -0.3249 - val_dice_coef: 0.3249
# Epoch 6/10
# 590/590 [==============================] - 149s - loss: -0.3172 - dice_coef: 0.3172 - val_loss: -0.3251 - val_dice_coef: 0.3251
# Epoch 7/10
# 590/590 [==============================] - 151s - loss: -0.3174 - dice_coef: 0.3174 - val_loss: -0.3253 - val_dice_coef: 0.3253
# Epoch 8/10
# 590/590 [==============================] - 166s - loss: -0.3176 - dice_coef: 0.3176 - val_loss: -0.3255 - val_dice_coef: 0.3255
# Epoch 9/10
# 590/590 [==============================] - 174s - loss: -0.3177 - dice_coef: 0.3177 - val_loss: -0.3257 - val_dice_coef: 0.3257
# Epoch 10/10
# 256/590 [============>.................] - ETA: 136s - loss: -0.3114 - dice_coef: 0.3114^CTraceback (most recent call last):


# Output for filter 32 without mean and scaling normalisation. Epoch 11 to 30
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ((2624, 3, 32, 32), (2624, 1, 32, 32), 164, 2624)
# (61, 61)
# ('Num images total, train, valid ', 2624, 2361, 263)
# Train on 2361 samples, validate on 263 samples
# Epoch 1/10
# 2361/2361 [==============================] - 125s - loss: -0.8560 - dice_coef: 0.8560 - val_loss: -0.8491 - val_dice_coef: 0.8491
# Epoch 2/10
# 2361/2361 [==============================] - 146s - loss: -0.8592 - dice_coef: 0.8592 - val_loss: -0.8627 - val_dice_coef: 0.8627
# Epoch 3/10
# 2361/2361 [==============================] - 164s - loss: -0.8659 - dice_coef: 0.8659 - val_loss: -0.8674 - val_dice_coef: 0.8674
# Epoch 4/10
# 2361/2361 [==============================] - 150s - loss: -0.8672 - dice_coef: 0.8672 - val_loss: -0.8591 - val_dice_coef: 0.8591
# Epoch 5/10
# 2361/2361 [==============================] - 135s - loss: -0.8705 - dice_coef: 0.8705 - val_loss: -0.8722 - val_dice_coef: 0.8722
# Epoch 6/10
# 2361/2361 [==============================] - 158s - loss: -0.8762 - dice_coef: 0.8762 - val_loss: -0.8701 - val_dice_coef: 0.8701
# Epoch 7/10
# 2361/2361 [==============================] - 138s - loss: -0.8785 - dice_coef: 0.8785 - val_loss: -0.8739 - val_dice_coef: 0.8739
# Epoch 8/10
# 2361/2361 [==============================] - 134s - loss: -0.8787 - dice_coef: 0.8787 - val_loss: -0.8693 - val_dice_coef: 0.8693
# Epoch 9/10
# 2361/2361 [==============================] - 141s - loss: -0.8813 - dice_coef: 0.8813 - val_loss: -0.8677 - val_dice_coef: 0.8677
# Epoch 10/10
# 2361/2361 [==============================] - 154s - loss: -0.8809 - dice_coef: 0.8809 - val_loss: -0.8810 - val_dice_coef: 0.8810
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ((2624, 3, 32, 32), (2624, 1, 32, 32), 164, 2624)
# (61, 61)
# ('Num images total, train, valid ', 2624, 2361, 263)
# Train on 2361 samples, validate on 263 samples
# Epoch 1/10
# 2361/2361 [==============================] - 170s - loss: -0.8823 - dice_coef: 0.8823 - val_loss: -0.8764 - val_dice_coef: 0.8764
# Epoch 2/10
# 2361/2361 [==============================] - 155s - loss: -0.8836 - dice_coef: 0.8836 - val_loss: -0.8839 - val_dice_coef: 0.8839
# Epoch 3/10
# 2361/2361 [==============================] - 166s - loss: -0.8847 - dice_coef: 0.8847 - val_loss: -0.8749 - val_dice_coef: 0.8749
# Epoch 4/10
# 2361/2361 [==============================] - 169s - loss: -0.8838 - dice_coef: 0.8838 - val_loss: -0.8871 - val_dice_coef: 0.8871
# Epoch 5/10
# 2361/2361 [==============================] - 164s - loss: -0.8849 - dice_coef: 0.8849 - val_loss: -0.8692 - val_dice_coef: 0.8692
# Epoch 6/10
# 2361/2361 [==============================] - 135s - loss: -0.8807 - dice_coef: 0.8807 - val_loss: -0.8769 - val_dice_coef: 0.8769
# Epoch 7/10
# 2361/2361 [==============================] - 166s - loss: -0.8888 - dice_coef: 0.8888 - val_loss: -0.8825 - val_dice_coef: 0.8825
# Epoch 8/10
# 2361/2361 [==============================] - 152s - loss: -0.8905 - dice_coef: 0.8905 - val_loss: -0.8784 - val_dice_coef: 0.8784
# Epoch 9/10
# 2361/2361 [==============================] - 174s - loss: -0.8910 - dice_coef: 0.8910 - val_loss: -0.8919 - val_dice_coef: 0.8919
# Epoch 10/10
# 2361/2361 [==============================] - 212s - loss: -0.8923 - dice_coef: 0.8923 - val_loss: -0.8869 - val_dice_coef: 0.8869
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ 



# Weights and performance of 32x32 filter Usegment: 0.82 in score
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ('Mean R, G, B, M', 185.16876637570655, 179.57174997424048, 185.93117342670553, 0.082145941278065626)
# ((6369, 3, 32, 32), (6369, 1, 32, 32), 169, 6369)
# ('Num images total, train, valid ', 6369, 5732, 637)
# ('Mean R, G, B', 184.41132499018667, 180.09310122993719, 186.02098927893843)
# Train on 5732 samples, validate on 637 samples
# Epoch 1/10
# 5732/5732 [==============================] - 406s - loss: -0.7892 - dice_coef: 0.7892 - val_loss: -0.8433 - val_dice_coef: 0.8433
# Epoch 2/10
# 5732/5732 [==============================] - 358s - loss: -0.8057 - dice_coef: 0.8057 - val_loss: -0.8193 - val_dice_coef: 0.8193
# Epoch 3/10
# 5732/5732 [==============================] - 307s - loss: -0.8182 - dice_coef: 0.8182 - val_loss: -0.8600 - val_dice_coef: 0.8600
# Epoch 4/10
# 5732/5732 [==============================] - 305s - loss: -0.8303 - dice_coef: 0.8303 - val_loss: -0.8580 - val_dice_coef: 0.8580
# Epoch 5/10
# 5732/5732 [==============================] - 272s - loss: -0.8352 - dice_coef: 0.8352 - val_loss: -0.8544 - val_dice_coef: 0.8544
# Epoch 6/10
# 5732/5732 [==============================] - 278s - loss: -0.8375 - dice_coef: 0.8375 - val_loss: -0.8643 - val_dice_coef: 0.8643
# Epoch 7/10
# 5732/5732 [==============================] - 281s - loss: -0.8489 - dice_coef: 0.8489 - val_loss: -0.8561 - val_dice_coef: 0.8561
# Epoch 8/10
# 5732/5732 [==============================] - 281s - loss: -0.8539 - dice_coef: 0.8539 - val_loss: -0.8535 - val_dice_coef: 0.8535
# Epoch 9/10
# 5732/5732 [==============================] - 267s - loss: -0.8537 - dice_coef: 0.8537 - val_loss: -0.8737 - val_dice_coef: 0.8737
# Epoch 10/10
# 5732/5732 [==============================] - 298s - loss: -0.8669 - dice_coef: 0.8669 - val_loss: -0.8487 - val_dice_coef: 0.8487
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ('Mean R, G, B, M', 185.16876637570655, 179.57174997424048, 185.93117342670553, 0.082145941278065626)
# ((6369, 3, 32, 32), (6369, 1, 32, 32), 169, 6369)
# ('Num images total, train, valid ', 6369, 5732, 637)
# ('Mean R, G, B', 184.41132499018667, 180.09310122993719, 186.02098927893843)
# Train on 5732 samples, validate on 637 samples
# Epoch 1/30
# 5732/5732 [==============================] - 279s - loss: -0.8458 - dice_coef: 0.8458 - val_loss: -0.8682 - val_dice_coef: 0.8682
# Epoch 2/30
# 5732/5732 [==============================] - 337s - loss: -0.8557 - dice_coef: 0.8557 - val_loss: -0.8766 - val_dice_coef: 0.8766
# Epoch 3/30
# 5732/5732 [==============================] - 334s - loss: -0.8546 - dice_coef: 0.8546 - val_loss: -0.8724 - val_dice_coef: 0.8724
# Epoch 4/30
# 5732/5732 [==============================] - 298s - loss: -0.8567 - dice_coef: 0.8567 - val_loss: -0.8735 - val_dice_coef: 0.8735
# Epoch 5/30
# 5732/5732 [==============================] - 309s - loss: -0.8546 - dice_coef: 0.8546 - val_loss: -0.8671 - val_dice_coef: 0.8671
# Epoch 6/30
# 5732/5732 [==============================] - 349s - loss: -0.8665 - dice_coef: 0.8665 - val_loss: -0.8820 - val_dice_coef: 0.8820
# Epoch 7/30
# 5732/5732 [==============================] - 406s - loss: -0.8629 - dice_coef: 0.8629 - val_loss: -0.8344 - val_dice_coef: 0.8344
# Epoch 8/30
# 5732/5732 [==============================] - 485s - loss: -0.8734 - dice_coef: 0.8734 - val_loss: -0.8821 - val_dice_coef: 0.8821
# Epoch 9/30
# 5732/5732 [==============================] - 641s - loss: -0.8791 - dice_coef: 0.8791 - val_loss: -0.8814 - val_dice_coef: 0.8814
# Epoch 10/30
# 5732/5732 [==============================] - 614s - loss: -0.8816 - dice_coef: 0.8816 - val_loss: -0.8789 - val_dice_coef: 0.8789
# Epoch 11/30
# 5732/5732 [==============================] - 646s - loss: -0.8843 - dice_coef: 0.8843 - val_loss: -0.8825 - val_dice_coef: 0.8825
# Epoch 12/30
# 5732/5732 [==============================] - 707s - loss: -0.8876 - dice_coef: 0.8876 - val_loss: -0.8878 - val_dice_coef: 0.8878
# Epoch 13/30
# 5732/5732 [==============================] - 724s - loss: -0.8864 - dice_coef: 0.8864 - val_loss: -0.8853 - val_dice_coef: 0.8853
# Epoch 14/30
# 5732/5732 [==============================] - 755s - loss: -0.8927 - dice_coef: 0.8927 - val_loss: -0.8767 - val_dice_coef: 0.8767
# Epoch 15/30
# 5732/5732 [==============================] - 693s - loss: -0.8876 - dice_coef: 0.8876 - val_loss: -0.8639 - val_dice_coef: 0.8639
# Epoch 16/30
# 5732/5732 [==============================] - 752s - loss: -0.8875 - dice_coef: 0.8875 - val_loss: -0.8898 - val_dice_coef: 0.8898
# Epoch 17/30
# 5732/5732 [==============================] - 769s - loss: -0.8962 - dice_coef: 0.8962 - val_loss: -0.8898 - val_dice_coef: 0.8898
# Epoch 18/30
# 5732/5732 [==============================] - 725s - loss: -0.8966 - dice_coef: 0.8966 - val_loss: -0.8923 - val_dice_coef: 0.8923
# Epoch 19/30
# 5732/5732 [==============================] - 695s - loss: -0.9002 - dice_coef: 0.9002 - val_loss: -0.8909 - val_dice_coef: 0.8909
# Epoch 20/30
# 5732/5732 [==============================] - 705s - loss: -0.9026 - dice_coef: 0.9026 - val_loss: -0.8877 - val_dice_coef: 0.8877
# Epoch 21/30
# 5732/5732 [==============================] - 706s - loss: -0.9037 - dice_coef: 0.9037 - val_loss: -0.8928 - val_dice_coef: 0.8928
# Epoch 22/30
# 5732/5732 [==============================] - 687s - loss: -0.9022 - dice_coef: 0.9022 - val_loss: -0.8953 - val_dice_coef: 0.8953
# Epoch 23/30
# 5732/5732 [==============================] - 654s - loss: -0.8940 - dice_coef: 0.8940 - val_loss: -0.8878 - val_dice_coef: 0.8878
# Epoch 24/30
# 5732/5732 [==============================] - 685s - loss: -0.9058 - dice_coef: 0.9058 - val_loss: -0.8917 - val_dice_coef: 0.8917
# Epoch 25/30
# 5732/5732 [==============================] - 691s - loss: -0.9014 - dice_coef: 0.9014 - val_loss: -0.8918 - val_dice_coef: 0.8918
# Epoch 26/30
# 5732/5732 [==============================] - 702s - loss: -0.9075 - dice_coef: 0.9075 - val_loss: -0.8790 - val_dice_coef: 0.8790
# Epoch 27/30
# 5732/5732 [==============================] - 692s - loss: -0.9062 - dice_coef: 0.9062 - val_loss: -0.8841 - val_dice_coef: 0.8841
# Epoch 28/30
# 5732/5732 [==============================] - 681s - loss: -0.9020 - dice_coef: 0.9020 - val_loss: -0.8801 - val_dice_coef: 0.8801
# Epoch 29/30
# 5732/5732 [==============================] - 661s - loss: -0.9022 - dice_coef: 0.9022 - val_loss: -0.8949 - val_dice_coef: 0.8949
# Epoch 30/30
# 5732/5732 [==============================] - 692s - loss: -0.9084 - dice_coef: 0.9084 - val_loss: -0.8959 - val_dice_coef: 0.8959



# performance for Weights weight_cache_32_1_70.h5, weight_cache_32_1_85.h5
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ('Mean R, G, B, M', 185.16876637570655, 179.57174997424048, 185.93117342670553, 0.082145941278065626)
# ((6369, 3, 32, 32), (6369, 1, 32, 32), 169, 6369)
# ('Num images total, train, valid ', 6369, 5732, 637)
# ('Mean R, G, B', 184.41132499018667, 180.09310122993719, 186.02098927893843)
# Train on 5732 samples, validate on 637 samples
# Epoch 1/30
# 5732/5732 [==============================] - 693s - loss: -0.9026 - dice_coef: 0.9026 - val_loss: -0.8957 - val_dice_coef: 0.8957
# Epoch 2/30
# 5732/5732 [==============================] - 684s - loss: -0.9056 - dice_coef: 0.9056 - val_loss: -0.8990 - val_dice_coef: 0.8990
# Epoch 3/30
# 5732/5732 [==============================] - 752s - loss: -0.9055 - dice_coef: 0.9055 - val_loss: -0.8874 - val_dice_coef: 0.8874
# Epoch 4/30
# 5732/5732 [==============================] - 939s - loss: -0.9070 - dice_coef: 0.9070 - val_loss: -0.8976 - val_dice_coef: 0.8976
# Epoch 5/30
# 5732/5732 [==============================] - 810s - loss: -0.9114 - dice_coef: 0.9114 - val_loss: -0.8953 - val_dice_coef: 0.8953
# Epoch 6/30
# 5732/5732 [==============================] - 787s - loss: -0.9112 - dice_coef: 0.9112 - val_loss: -0.8968 - val_dice_coef: 0.8968
# Epoch 7/30
# 5732/5732 [==============================] - 868s - loss: -0.9023 - dice_coef: 0.9023 - val_loss: -0.8998 - val_dice_coef: 0.8998
# Epoch 8/30
# 5732/5732 [==============================] - 832s - loss: -0.9137 - dice_coef: 0.9137 - val_loss: -0.8994 - val_dice_coef: 0.8994
# Epoch 9/30
# 5732/5732 [==============================] - 788s - loss: -0.9162 - dice_coef: 0.9162 - val_loss: -0.8931 - val_dice_coef: 0.8931
# Epoch 10/30
# 5732/5732 [==============================] - 841s - loss: -0.9130 - dice_coef: 0.9130 - val_loss: -0.9008 - val_dice_coef: 0.9008
# Epoch 11/30
# 5732/5732 [==============================] - 803s - loss: -0.9100 - dice_coef: 0.9100 - val_loss: -0.8954 - val_dice_coef: 0.8954
# Epoch 12/30
# 5732/5732 [==============================] - 870s - loss: -0.9111 - dice_coef: 0.9111 - val_loss: -0.8951 - val_dice_coef: 0.8951
# Epoch 13/30
# 5732/5732 [==============================] - 834s - loss: -0.9120 - dice_coef: 0.9120 - val_loss: -0.8964 - val_dice_coef: 0.8964
# Epoch 14/30
# 5732/5732 [==============================] - 889s - loss: -0.9132 - dice_coef: 0.9132 - val_loss: -0.9003 - val_dice_coef: 0.9003
# Epoch 15/30
# 5732/5732 [==============================] - 878s - loss: -0.9151 - dice_coef: 0.9151 - val_loss: -0.8946 - val_dice_coef: 0.8946
# Epoch 16/30
# 5732/5732 [==============================] - 913s - loss: -0.9136 - dice_coef: 0.9136 - val_loss: -0.9008 - val_dice_coef: 0.9008
# Epoch 17/30
# 5732/5732 [==============================] - 930s - loss: -0.9157 - dice_coef: 0.9157 - val_loss: -0.8987 - val_dice_coef: 0.8987
# Epoch 18/30
# 5732/5732 [==============================] - 877s - loss: -0.9101 - dice_coef: 0.9101 - val_loss: -0.8994 - val_dice_coef: 0.8994
# Epoch 19/30
# 5732/5732 [==============================] - 1135s - loss: -0.9080 - dice_coef: 0.9080 - val_loss: -0.8952 - val_dice_coef: 0.8952
# Epoch 20/30
# 5732/5732 [==============================] - 1009s - loss: -0.9133 - dice_coef: 0.9133 - val_loss: -0.8945 - val_dice_coef: 0.8945
# Epoch 21/30
# 5732/5732 [==============================] - 743s - loss: -0.9106 - dice_coef: 0.9106 - val_loss: -0.8996 - val_dice_coef: 0.8996
# Epoch 22/30
# 5732/5732 [==============================] - 697s - loss: -0.9150 - dice_coef: 0.9150 - val_loss: -0.9001 - val_dice_coef: 0.9001
# Epoch 23/30
# 5732/5732 [==============================] - 725s - loss: -0.9193 - dice_coef: 0.9193 - val_loss: -0.9042 - val_dice_coef: 0.9042
# Epoch 24/30
# 5732/5732 [==============================] - 726s - loss: -0.9210 - dice_coef: 0.9210 - val_loss: -0.9018 - val_dice_coef: 0.9018
# Epoch 25/30
# 5732/5732 [==============================] - 708s - loss: -0.9166 - dice_coef: 0.9166 - val_loss: -0.8938 - val_dice_coef: 0.8938
# Epoch 26/30
# 5732/5732 [==============================] - 717s - loss: -0.9161 - dice_coef: 0.9161 - val_loss: -0.8998 - val_dice_coef: 0.8998
# Epoch 27/30
# 5732/5732 [==============================] - 751s - loss: -0.9166 - dice_coef: 0.9166 - val_loss: -0.9035 - val_dice_coef: 0.9035
# Epoch 28/30
# 5732/5732 [==============================] - 1000s - loss: -0.9161 - dice_coef: 0.9161 - val_loss: -0.9004 - val_dice_coef: 0.9004
# Epoch 29/30
# 5732/5732 [==============================] - 783s - loss: -0.9162 - dice_coef: 0.9162 - val_loss: -0.9047 - val_dice_coef: 0.9047
# Epoch 30/30
# 5732/5732 [==============================] - 873s - loss: -0.9190 - dice_coef: 0.9190 - val_loss: -0.9023 - val_dice_coef: 0.9023
# Kannappans-MacBook-Pro:01_train_data kannappanjayakodinitthilan$ python train.py 
# Using Theano backend.
# ('Mean R, G, B, M', 185.16876637570655, 179.57174997424048, 185.93117342670553, 0.082145941278065626)
# ((6369, 3, 32, 32), (6369, 1, 32, 32), 169, 6369)
# ('Num images total, train, valid ', 6369, 5732, 637)
# ('Mean R, G, B', 184.41132499018667, 180.09310122993719, 186.02098927893843)
# Train on 5732 samples, validate on 637 samples
# Epoch 1/30
# 5732/5732 [==============================] - 861s - loss: -0.9177 - dice_coef: 0.9177 - val_loss: -0.8945 - val_dice_coef: 0.8945
# Epoch 2/30
# 5732/5732 [==============================] - 851s - loss: -0.9155 - dice_coef: 0.9155 - val_loss: -0.9053 - val_dice_coef: 0.9053
# Epoch 3/30
# 5732/5732 [==============================] - 892s - loss: -0.9156 - dice_coef: 0.9156 - val_loss: -0.9051 - val_dice_coef: 0.9051
# Epoch 4/30
# 5732/5732 [==============================] - 958s - loss: -0.9199 - dice_coef: 0.9199 - val_loss: -0.9048 - val_dice_coef: 0.9048
# Epoch 5/30
# 5732/5732 [==============================] - 1252s - loss: -0.9183 - dice_coef: 0.9183 - val_loss: -0.9063 - val_dice_coef: 0.9063
# Epoch 6/30
# 5732/5732 [==============================] - 905s - loss: -0.9224 - dice_coef: 0.9224 - val_loss: -0.8974 - val_dice_coef: 0.8974
# Epoch 7/30
# 5732/5732 [==============================] - 866s - loss: -0.9211 - dice_coef: 0.9211 - val_loss: -0.9049 - val_dice_coef: 0.9049
# Epoch 8/30
# 5732/5732 [==============================] - 890s - loss: -0.9203 - dice_coef: 0.9203 - val_loss: -0.9018 - val_dice_coef: 0.9018
# Epoch 9/30
# 5732/5732 [==============================] - 866s - loss: -0.9226 - dice_coef: 0.9226 - val_loss: -0.9000 - val_dice_coef: 0.9000
# Epoch 10/30
# 5732/5732 [==============================] - 849s - loss: -0.9195 - dice_coef: 0.9195 - val_loss: -0.8956 - val_dice_coef: 0.8956
# Epoch 11/30
# 5732/5732 [==============================] - 800s - loss: -0.9150 - dice_coef: 0.9150 - val_loss: -0.9022 - val_dice_coef: 0.9022
# Epoch 12/30
# 5732/5732 [==============================] - 769s - loss: -0.9196 - dice_coef: 0.9196 - val_loss: -0.9052 - val_dice_coef: 0.9052
# Epoch 13/30
# 5732/5732 [==============================] - 869s - loss: -0.9253 - dice_coef: 0.9253 - val_loss: -0.9072 - val_dice_coef: 0.9072
# Epoch 14/30
# 5732/5732 [==============================] - 770s - loss: -0.9241 - dice_coef: 0.9241 - val_loss: -0.9055 - val_dice_coef: 0.9055
# Epoch 15/30
# 5732/5732 [==============================] - 885s - loss: -0.9227 - dice_coef: 0.9227 - val_loss: -0.8928 - val_dice_coef: 0.8928
# Epoch 16/30
# 1408/5732 [======>.......................] - ETA: 654s - loss: -0.9133 - dice_coef: 0.9133^CTraceback (most recent call last):
#   File "train.py", line 92, in <module>
