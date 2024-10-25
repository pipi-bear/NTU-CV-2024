import cv2
import numpy as np
import matplotlib.pyplot as plt

# aim: binarize the image

def main():
    # funct def: the imread function returns a numpy array, and the second parameter is the flag for color or grayscale
    # funct def: here we choose to let the flag to be 0, in order to get a grayscale image
    # explain: which means the value of each pixel is a value between 0 to 255, representing the intensity of the grayscale level 
    orig_lena = cv2.imread('lena.bmp', 0)
    height = orig_lena.shape[0]
    width = orig_lena.shape[1]

    # aim: get the binariazed numpy array of the image
    # explain: initialize the bin_lena to be a zero matrix with the same shape as the orig_lena,
    # explain: then only pixels with intensity value larger than 128 will be set to 255, otherwise 0
    bin_lena = np.zeros(orig_lena.shape, dtype = np.uint8)      
    
    for i in range(width):
        for j in range(height):
            if orig_lena[i][j] >= 128:
                bin_lena[i][j] = 255

    # aim: save the binarized image to a file named bin_lena.bmp
    cv2.imwrite('bin_lena.bmp', bin_lena)

    # aim: create the histogram of the original grayscale image
    # explain: the histogram is a numpy array with 256 elements, where each element represents the number of pixels with intensity value i, i = 0,...,255
    hist_orig_lena = np.zeros(256, dtype = np.uint32)

    for i in range(width):
        for j in range(height):
            hist_orig_lena[orig_lena[i][j]] += 1

    # aim: plot the histogram
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of the Original Grayscale Image')
    # funct def: the first parameter of plt.bar is the x-coordinate of the bars, which should be array-like (or float)
    # funct def: the second parameter of plt.bar is the height of the bars, which also should be array-like (or float)
    # explain: np.arange(256) generates an array of 256 elements, which are the intensity values 0 to 255
    # explain: hist_orig_lena is the height of the bars
    plt.bar(np.arange(256), hist_orig_lena)
    plt.show()

    # aim: calculate the connected components of the binarized image using the iterative algorithm
    # subaim: initialize each pixel to a unique label
    # explain: start from an array with all zeros, then update the array by changing the pixels with intensity value 255 to unique labels
    uniq_label_arr = np.zeros(orig_lena.shape, dtype = np.uint32)
    uniq_label = 0

    for i in range(width):
        for j in range(height):
            if bin_lena[i][j] > 0:
                uniq_label_arr[i][j] = uniq_label
                uniq_label += 1

    # subaim: iteration of top-down followed by bottom-up passes, until no label change happens
    # explain: this process is of two major parts, the first part is top-down pass, and the second part is bottom-up pass
    # explain: for top-down part, we have 9 cases to check for the type of each pixel
    # explain: the first four cases are for the pixels on the corners, the next four cases are for the pixels on the edges, and the last case is for the pixels in the middle of the image

    change = True

    while change:
        change = False

        # subsubaim: top-down pass
        for i in range(width):
            for j in range(height):
                # check for label change if the pixel is not of label 0
                if uniq_label_arr[i][j] != 0:
                    min_label = uniq_label_arr[i][j]

                    # case 1: the pixel is on the upper left corner
                    if i == 0 and j == 0:
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]
                        
                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 2: the pixel is on the upper right corner
                    elif i == 0 and j == width - 1:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 3: the pixel is on the lower left corner
                    elif i == height - 1 and j == 0:
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                    # case 4: the pixel is on the lower right corner
                    elif i == height - 1 and j == width - 1:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                    # case 5: the pixel is on the upper edge
                    elif i == 0:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 6: the pixel is on the lower edge
                    elif i == height - 1:   
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                    # case 7: the pixel is on the left edge
                    elif j == 0:
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above   
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 8: the pixel is on the right edge
                    elif j == width - 1:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above   
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    
                    # case 9: the pixel is in the middle of the image
                    else:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label

        # subsubaim: bottom-up pass
        # explain: the bottom-up pass is similar to the top-down pass, but it starts from the bottom right corner and goes to the top left corner
        # explain: thus we start from the last row (height-1) and the last column (width-1), and go to the first row (0) and the first column (0)

        for i in range(height-1, -1, -1):                           
            for j in range(width-1, -1, -1):
                if uniq_label_arr[i][j] != 0:
                    min_label = uniq_label_arr[i][j]

                    # case 1: the pixel is on the upper left corner
                    if i == 0 and j == 0:
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]
                        
                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 2: the pixel is on the upper right corner
                    elif i == 0 and j == width - 1:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 3: the pixel is on the lower left corner
                    elif i == height - 1 and j == 0:
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                    # case 4: the pixel is on the lower right corner
                    elif i == height - 1 and j == width - 1:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                    # case 5: the pixel is on the upper edge
                    elif i == 0:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 6: the pixel is on the lower edge
                    elif i == height - 1:   
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                    # case 7: the pixel is on the left edge
                    elif j == 0:
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above   
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    # case 8: the pixel is on the right edge
                    elif j == width - 1:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above   
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label
                    
                    # case 9: the pixel is in the middle of the image
                    else:
                        if uniq_label_arr[i][j-1] != 0 and uniq_label_arr[i][j-1] < min_label:      # check the pixel on its left
                            min_label = uniq_label_arr[i][j-1]
                        if uniq_label_arr[i-1][j] != 0 and uniq_label_arr[i-1][j] < min_label:      # check the pixel above
                            min_label = uniq_label_arr[i-1][j]
                        if uniq_label_arr[i][j+1] != 0 and uniq_label_arr[i][j+1] < min_label:      # check the pixel on its right
                            min_label = uniq_label_arr[i][j+1]
                        if uniq_label_arr[i+1][j] != 0 and uniq_label_arr[i+1][j] < min_label:      # check the pixel below
                            min_label = uniq_label_arr[i+1][j]

                        if uniq_label_arr[i][j] != min_label:
                            change = True

                        # assign the minimum label to the neighbors of the current pixel
                        if uniq_label_arr[i][j-1] != 0:             # assign the minimum label to the pixel on its left
                            uniq_label_arr[i][j-1] = min_label
                        if uniq_label_arr[i-1][j] != 0:             # assign the minimum label to the pixel above
                            uniq_label_arr[i-1][j] = min_label
                        if uniq_label_arr[i][j+1] != 0:             # assign the minimum label to the pixel on its right
                            uniq_label_arr[i][j+1] = min_label
                        if uniq_label_arr[i+1][j] != 0:             # assign the minimum label to the pixel below
                            uniq_label_arr[i+1][j] = min_label

    # subaim: count the number of each labels after updating the connected components with same labels
    # note: we use 500 pixels as a threshold, so regions that have pixel count less than 500 are omitted

    pixel_count_arr = np.zeros(np.max(uniq_label_arr)+1, dtype = np.uint32)
    
    # subsubaim: count the number of each labels
    for i in range(height):
        for j in range(width):
            if uniq_label_arr[i][j] != 0:
                pixel_count_arr[uniq_label_arr[i][j]] += 1

    # subsubaim: store the labels that have pixel count more than 500 to target_label_arr
    target_label_arr = []
    for i in range(len(pixel_count_arr)):
        if pixel_count_arr[i] >= 500: 
            target_label_arr.append(i)

    # note: the true region amount given by the TA is 5(which is the same as the result of the following line)
    region_amount = len(target_label_arr)

    # subaim: plot the rectangles of the regions
    # subsubaim: find the minimum and maximum x and y coordinates of the region, and store them in the region_locations list
    region_locations = []
    for i in range(len(target_label_arr)):
        rectangle = {}

        # explain: the np.where function returns the indices of the elements that satisfy the condition (returns an ndarray)
        # explain: we then use the min and max functions to find the minimum and maximum x and y coordinates from those indicies
        min_x = np.min(np.where(uniq_label_arr == target_label_arr[i])[0])
        max_x = np.max(np.where(uniq_label_arr == target_label_arr[i])[0])
        min_y = np.min(np.where(uniq_label_arr == target_label_arr[i])[1])
        max_y = np.max(np.where(uniq_label_arr == target_label_arr[i])[1])
        
        rectangle['x_1'] = min_x
        rectangle['x_2'] = max_x
        rectangle['y_1'] = min_y
        rectangle['y_2'] = max_y

        region_locations.append(rectangle)

    # subsubaim: Create a 3-channel BGR image based on the binarized lena image, so that we can draw colored rectangles on it
    rectangle_image = cv2.cvtColor(bin_lena, cv2.COLOR_GRAY2BGR)

    # subsubaim: add each rectangle on the image
    for rectangles in region_locations:
        x1, x2, y1, y2 = rectangles['x_1'], rectangles['x_2'], rectangles['y_1'], rectangles['y_2']
        # funct def: cv2.rectangle(image, start_point, end_point, color, thickness)
        # funct def: the parameter start_point is the coordinate represented as tuples of two values (i.e. x-coordinate value, y-coordinate value)
        # note: openCV uses (y, x) format, so we need to swap x1, x2, y1, y2 when drawing the rectangle
        cv2.rectangle(rectangle_image, (y1, x1), (y2, x2), (0, 0, 255), 2)
    
    # subsubaim: Save the image with rectangles
    cv2.imwrite('bin_lena_with_rectangles.bmp', rectangle_image)

    # subsubaim: create a copy of the rectangle image, so that we can add centroids on it
    centroid_image = rectangle_image.copy()

    # subsubaim: calculate the centroids of the regions
    centroid_locations = []
    for i in range(len(target_label_arr)):
        # get all the coordinates where the label matches the current target label
        y_coords, x_coords = np.where(uniq_label_arr == target_label_arr[i])
        
        # calculate the total number of pixels in this region
        total_pixels = len(x_coords)
        
        # sum up all x and y coordinates
        sum_x = 0
        sum_y = 0
        for i in range(total_pixels):
            sum_x += x_coords[i]
            sum_y += y_coords[i]
        
        # calculate the centroid coordinates
        mean_x = int(sum_x / total_pixels)
        mean_y = int(sum_y / total_pixels)

        centroid_locations.append((mean_x, mean_y))

    # subsubaim: add each centroid on the image
    for mean_x, mean_y in centroid_locations:
        # funct def: cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(centroid_image, (mean_x, mean_y), 2, (255, 0, 0), 2)
    
    # subsubaim: Save the image with centroids
    cv2.imwrite('bin_lena_with_centroids_and_rectangles.bmp', centroid_image)
    cv2.imshow('Lena with Rectangles and Centroids', centroid_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()