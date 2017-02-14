import data_loading as dl;
import cv2
import numpy as np

DATA_CACHE_BASE_PATH = "../../../data/sig_tuple_seg/cache/"
TEST_OUTPUT_BASE_PATH = DATA_CACHE_BASE_PATH+"output_64_1_17_ovr_lpd_25/"
PP_OUTPUT_BASE_PATH = DATA_CACHE_BASE_PATH+"output_64_1_17_ovr_lpd_pp_25/"
DATA_BASE_PATH = "../../../data/sig_tuple_seg/SigTuple_data/"
TEST_PATH = DATA_BASE_PATH+"Test_Data/"



(test_list, test_file_list) = dl.get_test_images(TEST_OUTPUT_BASE_PATH, 3)

print("test list ", len(test_list), test_file_list[0], test_list[0].shape)

num_image = 0
for img in test_list:
	# kernel = np.ones((8,8),np.uint8)
	# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	file_path = TEST_PATH+test_file_list[num_image][:-9]+".jpg"
	org_img = cv2.imread(file_path)
	print("File Info ", file_path, test_file_list[num_image])
	print("Image info ", org_img.shape, img.shape)
	img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(img1,127,255,0)

	mask_img = np.zeros(org_img.shape)
	(contours, hierarchy) = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print (PP_OUTPUT_BASE_PATH+test_file_list[num_image], len(contours))
	#cv2.imwrite(PP_OUTPUT_BASE_PATH+test_file_list[num_image], opening);

	hull_approx = []
	contour_reduced = []
	for cnt in contours:
		M = cv2.moments(cnt)
		area = cv2.contourArea(cnt)
		perimeter = cv2.arcLength(cnt,True)
		if(area > 750):
			print("Area, perimeter, moment ", area, perimeter, 
				int(M['m10']/M['m00']), int(M['m01']/M['m00']))
			# epsilon = 0.01*cv2.arcLength(cnt,True)
			# approx = cv2.approxPolyDP(cnt,epsilon,True)
			# # print ("Contours ", cnt, approx)
			# #approx = np.int0(approx)
			# reduced_contours.append(approx)

			hull = cv2.convexHull(cnt)
			hull_area = cv2.contourArea(hull)
			hull_approx.append(hull)

			ellipse = cv2.fitEllipse(cnt)
			#cv2.ellipse(org_img, ellipse,(0,0,255),2)

			#cv2.ellipse(mask_img, ellipse, (255, 255, 255), -1)

			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			box_area = cv2.contourArea(box)

			print("Contour/Box area ", box_area, 
				hull_area, hull_area/box_area)

			(x,y),radius = cv2.minEnclosingCircle(cnt)
			center = (int(x),int(y))
			radius = int(radius)
			#cv2.circle(org_img,center,radius,(255,0,0),2)

			contour_reduced.append(cnt)



	#cv2.drawContours(org_img, hull_approx, -1, (0,255,0), 2)
	#cv2.drawContours(img, ellipse_approx, -1, (255,0,0), 3)
	#cv2.imshow('Contours',mask_img)
	#cv2.drawContours(org_img, contour_reduced, -1, (0,0,255), 1)
	cv2.drawContours(mask_img, contour_reduced, -1, (255,255,255), -1)
	# kernel = np.ones((3,3),np.uint8)
	# mask_img = cv2.dilate(mask_img,kernel,iterations = 1)

	cv2.imwrite(PP_OUTPUT_BASE_PATH+test_file_list[num_image], mask_img);


	# k = cv2.waitKey(5) & 0xFF
	# while True:
	# 	if cv2.waitKey(6) & 0xff == 27:
	# 		break
	num_image += 1
cv2.destroyAllWindows()

# for img in test_list:
# 	print("shape ", img.shape)
# 	#img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 	ret,thresh = cv2.threshold(img,127,255,0)
# 	(contours, hierarchy) = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# 	#print "Num contours ", len(contours), contours[0].shape
# 	# for data in contours:
# 	# 	print "The contours have this data %r" %data
# 	cv2.drawContours(img, contours, -1, (0,255,0), 3)
# 	cv2.imshow('Contours',img)
# 	k = cv2.waitKey(5) & 0xFF
# 	while True:
# 		if cv2.waitKey(6) & 0xff == 27:
# 			break
# cv2.destroyAllWindows()


