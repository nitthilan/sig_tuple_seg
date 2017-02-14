from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as mpplt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import matplotlib.patches as patches
import pickle
import mpld3




def get_img_files(base_dir):
	files = [join(base_dir, f) for f in listdir(base_dir) if (isfile(join(base_dir, f)) \
		and f.lower().endswith('.jpg'))]

	# print(len(files), files[0])
	return files;

def get_train_files(train_base_dir):
	train_dir_list = [join(train_base_dir, f) for f in listdir(train_base_dir) if not isfile(join(train_base_dir, f))]
	train_image_list = []
	for train_dir in train_dir_list:
		train_img_list = get_img_files(train_dir)
		train_image_list.extend(train_img_list)
		# print(len(train_image_list), len(train_img_list), train_dir, train_img_list[0])
	return train_image_list

class FileDatabase:
	def __init__(self, image_list, rect_info_path):
		self.image_list = image_list
		self.rect_info_path = rect_info_path
		self.rect_info = {}
		self.skip_marked_images = True;
		self.img_idx = 0;

	def load_rect_info_file(self):
		try:
			with open(self.rect_info_path, 'rb') as f:
				self.rect_info = pickle.load(f)
		except EnvironmentError:
			print("File not present ", self.rect_info_path)
		return
	
	def store_rect_info_file(self):
		with open(self.rect_info_path, 'wb') as f:
			pickle.dump(self.rect_info, f, pickle.HIGHEST_PROTOCOL)
		return

	def set_rect_info(self, coordinates):
		filename = self.image_list[self.img_idx]
		self.rect_info[filename] = (filename, coordinates)

	def get_rect_info(self):
		filename = self.image_list[self.img_idx]
		if(filename in self.rect_info):
			return self.rect_info[filename][1]
		else:
			return []

	def check_marked(self):
		if(not self.skip_marked_images):
			return False
		if('/NoF' in self.image_list[self.img_idx]):
			print "Has NoF"
			return True
		if(not self.image_list[self.img_idx] in self.rect_info):
			return False
		elif(len(self.rect_info[self.image_list[self.img_idx]][1])):
			return True
		else:
			return False

	def get_next_image(self):
		self.img_idx = self.img_idx + 1
		if(self.img_idx == len(self.image_list)):
			self.img_idx = 0
		while(self.check_marked()):
			print("Next image ", self.image_list[self.img_idx])
			self.img_idx = self.img_idx + 1
			if(self.img_idx == len(self.image_list)):
				self.img_idx = 0
		return self.image_list[self.img_idx]

	def get_prev_image(self):
		self.img_idx = self.img_idx - 1
		if(self.img_idx == -1):
			self.img_idx = len(self.image_list) - 1
		while(self.check_marked()):
			print("Prev image ", self.image_list[self.img_idx])
			self.img_idx = self.img_idx - 1
			if(self.img_idx == -1):
				self.img_idx = len(self.image_list) - 1
		return self.image_list[self.img_idx]

	def get_skip_marked_images_flag(self):
		return self.skip_marked_images

	def set_skip_marked_images_flag(self, skip_marked_images):
		self.skip_marked_images = skip_marked_images


class FileHandler:
	def __init__(self, test_base_dir, test_store_path):
		self.test_base_dir = test_base_dir;

		self.test_image_db = FileDatabase(get_img_files(test_base_dir), test_store_path)
		#self.train_image_db = FileDatabase(get_train_files(train_base_dir), train_store_path)
		#self.is_train_set = True;
		self.image_db = self.test_image_db

		self.train_rect_info = {}
		self.test_rect_info = {}
		return

	def save_rect_info_event():
		return


	# def get_is_train_set_flag(self):
	# 	return self.is_train_set;

	# def set_is_train_set_flag(self, is_train_set):
	# 	self.is_train_set = is_train_set
	# 	if(is_train_set):
	# 		self.image_db = self.train_image_db
	# 	else:
	# 		self.image_db = self.test_image_db
	# 	return

class ImgDisplay:
	def __init__(self, file_handler):

		# Initialise the figures
		self.fig = mpplt.figure()
		mpplt.subplots_adjust(left=0, bottom=0.05, right=1, top=1, wspace=None, hspace=None)
		self.ax = self.fig.add_subplot(111, aspect='equal')
		
		#Initialise event handlers
		ni = self.add_button([0.7 , 0.05, 0.04, 0.04], 'NI', self.next_image_event) # Select prev image
		pi = self.add_button([0.75, 0.05, 0.04, 0.04], 'PI', self.prev_image_event) # Select next image
		st = self.add_button([0.8 , 0.05, 0.04, 0.04], 'ST', self.store_rect_info_event) # Save the collected information
		#sw = self.add_button([0.85, 0.05, 0.04, 0.04], 'SW', self.switch_train_test_event) # Select between test and train sets
		sk = self.add_button([0.9 , 0.05, 0.04, 0.04], 'SK', self.skip_marked_images_event) # Select between test and train sets
		ps = self.add_button([0.95, 0.05, 0.04, 0.04], 'PS', self.print_state) # Print the state
		cl = self.add_button([0.65, 0.05, 0.04, 0.04], 'CI', self.clear_image) # clear all rectangle boxes
		ld = self.add_button([0.6 , 0.05, 0.04, 0.04], 'LD', self.load_rect_info_event) # load already created values

		cid1 = self.fig.canvas.mpl_connect('button_press_event'  , self.button_press_event)
		cid2 = self.fig.canvas.mpl_connect('button_release_event', self.button_release_event)

		self.filename = file_handler.image_db.get_next_image()
		self.ax.imshow(mpimg.imread(self.filename))

		# Initialise state variables
		self.file_handler = file_handler

		# Initialise the text field
		self.is_train_set_state = None
		self.skip_marked_images_state = None
		# self.write_text()

		# initialise 
		self.file_handler = file_handler
		# coordinate information
		self.coordinates = self.file_handler.image_db.get_rect_info() #Store the clicked coordinates
		
	
		mpplt.show()
		#mpld3.show()


	def add_button(self, button_pos, button_text, event_handler):
		axprev = mpplt.axes(button_pos)
		bprev = Button(axprev, button_text)
		bprev.on_clicked(event_handler)
		return bprev

	def write_text(self):
		if(self.is_train_set_state):
			self.is_train_set_state.set_visible(False)
			self.is_train_set_state.remove();
			del self.is_train_set_state
		if(self.skip_marked_images_state):
			self.skip_marked_images_state.remove()
		if(self.file_handler.get_is_train_set_flag()):
			is_train_set_string = "Train Set"
		else:
			is_train_set_string = "Test Set"
		self.is_train_set_state = mpplt.text(.9, .9, is_train_set_string, horizontalalignment='left',
			verticalalignment='bottom', fontsize=10, transform=self.ax.transAxes)
		if(self.file_handler.get_skip_marked_images_flag()):
			skip_marked_images_string = "Skip Marked Images"
		else:
			skip_marked_images_string = "Review All Images"
		self.is_train_set_state = mpplt.text(.9, .8, skip_marked_images_string, horizontalalignment='left',
        	verticalalignment='bottom', fontsize=10, transform=self.ax.transAxes)
		mpplt.draw()

	def clear_image(self, event):
		self.coordinates = [];
		self.ax.clear()
		self.ax.imshow(mpimg.imread(self.filename)); mpplt.draw()
		return



	def button_press_event(self, event):
		if(event.xdata == None or event.ydata == None or event.xdata <= 1 or event.ydata <= 1):
			return
		self.click0 = (event.xdata, event.ydata)
		return

	def button_release_event(self, event):
		if(event.xdata == None or event.ydata == None or event.xdata <= 1 or event.ydata <= 1):
			return
		self.click1 = (event.xdata, event.ydata)
		if(self.click0[0] < self.click1[0]): x0 = self.click0[0]; x1 = self.click1[0];
		else: x0 = self.click1[0]; x1 = self.click0[0];
		if(self.click0[1] < self.click1[1]): y0 = self.click0[1]; y1 = self.click1[1];
		else: y0 = self.click1[1]; y1 = self.click0[1];
		self.coordinates.append((x0,y0,x1-x0,y1-y0))
		print("Image clicked ", (x0,y0,x1-x0,y1-y0), len(self.coordinates))
		for coord in self.coordinates:
			p = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], alpha=1, fill=False)
			self.ax.add_patch(p)
		mpplt.draw()
		return

	def change_image(self, is_next):
		self.file_handler.image_db.set_rect_info(self.coordinates)
		self.clear_image(None);

		if(is_next):
			self.filename = self.file_handler.image_db.get_next_image()
		else:
			self.filename = self.file_handler.image_db.get_prev_image()
		self.coordinates = self.file_handler.image_db.get_rect_info()
		#print(self.coordinates)
		for coord in self.coordinates:
			p = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], alpha=1, fill=False)
			self.ax.add_patch(p)
		self.ax.imshow(mpimg.imread(self.filename)); mpplt.draw()
		return;

	def next_image_event(self, event):
		self.change_image(True)
		

	def prev_image_event(self, event):
		self.change_image(False)

	def store_rect_info_event(self, event):
		print("Saved rect info")
		self.file_handler.image_db.store_rect_info_file();

	def load_rect_info_event(self, event):
		print("Loaded rect info")
		self.file_handler.image_db.load_rect_info_file();
		self.coordinates = self.file_handler.image_db.get_rect_info()
		for coord in self.coordinates:
			p = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], alpha=1, fill=False)
			self.ax.add_patch(p)
		self.ax.imshow(mpimg.imread(self.filename)); mpplt.draw()
		return
	
	# def switch_train_test_event(self, event):
	# 	self.file_handler.set_is_train_set_flag(not self.file_handler.get_is_train_set_flag())
	# 	self.filename = self.file_handler.image_db.get_next_image()
	# 	self.ax.imshow(mpimg.imread(self.filename)); mpplt.draw()
		
		# self.write_text()
		return;

	def print_state(self, event):
		print("Filename ", self.filename)
		print("Image Set Train ", self.file_handler.get_is_train_set_flag(), 
			"Skip images ", self.file_handler.image_db.get_skip_marked_images_flag(),
			"Total Images", len(self.file_handler.image_db.image_list),
			"Marked Images", len(self.file_handler.image_db.rect_info))

	def skip_marked_images_event(self, event):
		self.file_handler.image_db.set_skip_marked_images_flag(
			not self.file_handler.image_db.get_skip_marked_images_flag())
		return


def add_button(button_pos, button_text, event_handler):
	axprev = mpplt.axes(button_pos)
	bprev = Button(axprev, button_text)
	bprev.on_clicked(previous)
	return bprev

def previous(event):
	print("Previous", event);

	return


def main():	
	DATA_BASE_PATH = "../../../../data/sig_tuple_seg/SigTuple_data_new/"
	DATA_BASE_CACHE_PATH = "../../../data/sig_tuple_seg/cache/"
	TEST_DIR_PATH = DATA_BASE_PATH+"Test_Data/"
	TEST_STORE_PATH = DATA_BASE_CACHE_PATH+"test_mask.pkl"

	file_handler = FileHandler(TEST_DIR_PATH, TEST_STORE_PATH)
	img_display = ImgDisplay(file_handler)


if __name__ == "__main__":
    main()