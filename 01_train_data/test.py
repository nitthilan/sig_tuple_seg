
import data_loading as dl
import models as mo


DATA_BASE_PATH = "../../../data/sig_tuple_seg/SigTuple_data/"
DATA_CACHE_BASE_PATH = "../../../data/sig_tuple_seg/cache/"

TRAIN_PATH = DATA_BASE_PATH+"Train_Data/"
TEST_PATH = DATA_BASE_PATH+"Test_Data/"
TEST_CACHE_PATH = DATA_CACHE_BASE_PATH+"test.h5"
TRAIN_CACHE_PATH = DATA_CACHE_BASE_PATH+"train.h5"
#WEIGHT_CACHE_PATH = DATA_CACHE_BASE_PATH+"weight_cache_32_20.h5"
WEIGHT_CACHE_PATH = DATA_CACHE_BASE_PATH+"weight_cache_64_1_25.h5"

n_folds = 10;
random_state = 51
dimension = 128
num_color_component = 3
downscale_factor = 2
train_factor = 0.9
batch_size = 128
nb_epoch = 10

TEST_OUTPUT_BASE_PATH = DATA_CACHE_BASE_PATH+"output_64_1_17_ovr_lpd_25/"


#[test_list, test_file_list] = dl.get_test_images(TEST_PATH)
scaled_dimension = dimension/downscale_factor
#[test_scaled, test_file_list, test_file_dimension] = dl.get_test_images_dwn_scled(TEST_PATH, scaled_dimension)
[test_scaled, test_file_list, test_file_dimension] = dl.get_test_images_ovr_lap(TEST_PATH, scaled_dimension)
#[test_scaled, test_file_list, test_file_dimension] = dl.get_test_images_ovr_lap_off(TEST_PATH, scaled_dimension)

print("Read test images", len(test_scaled), test_scaled[0].shape)

model = mo.get_unet(num_color_component, scaled_dimension);

model.load_weights(WEIGHT_CACHE_PATH)
print("Model Loaded")

test_y = model.predict(test_scaled, batch_size=batch_size, verbose=1)
print("Prediction Ends ", test_y.shape)

#dl.dump_test_images_dwn_scled(TEST_OUTPUT_BASE_PATH, test_file_list, test_file_dimension, test_y, scaled_dimension)#, TEST_PATH)
dl.dump_test_images_ovr_lpd(TEST_OUTPUT_BASE_PATH, test_file_list, test_file_dimension, test_y, scaled_dimension)#, TEST_PATH)
#dl.dump_test_images_ovr_lpd_off(TEST_OUTPUT_BASE_PATH, test_file_list, test_file_dimension, test_y, scaled_dimension)#, TEST_PATH)
print("Test image dumped")



