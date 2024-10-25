import cv2
import numpy as np


# aim: define the path of the lena img
lena_img_path = "/Users/luo-chunchou/graduate stuff/enrolled courses/CV/hw0/lena.bmp"

# aim: read the lena image and save the image as a ndarray in 'orig_lena_img'
# funct def: imread('filename',IMREAD_COLOR_BGR)
# explain: the output of imread is a ndarray
orig_lena_img = cv2.imread(lena_img_path)

# aim: get the xyz - axis value of the original lena img
# explain: even though the original img is greyscale, it is of 3 dimensions (the 3rd dimension represents RGB intensity values)
# explain: this is because sometimes when the img is saved in jpeg format, or because openCV defaults treating even grayscale images as 3-channel images 

orig_lena_img_shape = orig_lena_img.shape
img_width = orig_lena_img_shape[0]
img_height = orig_lena_img_shape[1]
channels = orig_lena_img_shape[2]

# aim: PART1: generate 3 transformed .bmp files of lena
# subaim: (a) upside-down lena

# initialize the resulting img by a blank ndarray (causing a black img, size as the orig lena img)
# explain: the np.zeros() funct create a float array by default but openCV expects image to be unit8 format, 
# explain: so not setting the correct dtype will result in a white img.
upside_down_res = np.zeros(orig_lena_img_shape, dtype = np.uint8)

for i in range(img_width):
    upside_down_res[i, :] = orig_lena_img[img_width - i - 1, :]

# funct def: cv2.imshow('window_name', image)
cv2.imshow('upside-down lena', upside_down_res)
cv2.waitKey(0)

# subaim: (b) right-side-left lena

right_side_left_res = np.zeros(orig_lena_img_shape, dtype = np.uint8)
for j in range(img_height):
    right_side_left_res[:, j] = orig_lena_img[:, img_height - j - 1]

cv2.imshow('right-side-left lena', right_side_left_res)
cv2.waitKey(0)

# subaim: (c) diagonally flip lena
# explain: first perform a left-right flip to the img, then rotate to make it diagonal
diag_flip_res = np.zeros(orig_lena_img_shape, dtype = np.uint8)
for j in range(img_height):
    diag_flip_res[:, j] = upside_down_res[:, img_height - j - 1]

cv2.imshow('diagonally flip lena', diag_flip_res)
cv2.waitKey(0)

# aim: PART2: generate 3 transformed .bmp files of lena
# subaim: (d) rotate 45 degrees clockwise lena

rotate_res = np.zeros((img_height, img_width, channels), dtype = np.uint8)
for i in range(img_width):
    for j in range(img_height):
        rotate_res[j][img_width - i - 1] = orig_lena_img[i][j]

cv2.imshow('rotate 45 degrees clockwise lena', rotate_res)
cv2.waitKey(0)

# subaim: (e) shrink in half (this part is performed by other application as stated in the report)

# subaim: (f) binarize at 128 to get a binary image

binarized_res = np.zeros(orig_lena_img_shape, dtype = np.uint8)

for i in range(img_width):
    for j in range(img_height):
        for k in range(channels):
            if orig_lena_img[i][j][k] > 128:
                binarized_res[i][j][k] = 255
            else:
                binarized_res[i][j][k] = 0

cv2.imshow('binarized lena', binarized_res)
cv2.waitKey(0)

# stay the window open
# explain: '0' represents keeping the window stay until a key is pressed
cv2.destroyAllWindows()