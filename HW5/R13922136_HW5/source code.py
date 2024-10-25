import numpy as np
import cv2

# aim: (a) dilation
# explain: grayscale dilation assign each pixel in the image with the maximum value of the pixels covered by the kernel
def dilation(image, kernel):
    dilation_image  = np.zeros(image.shape, np.uint8)                       # initialize the dilation image with zeros of the same size as the original image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            max_value = 0
            for k in range(len(kernel)):                                    # iterate through the elements in the kernel
                s, t = kernel[k][0] + i, kernel[k][1] + j                   # assign (the coordinates of the current element in the kernel + the current pixel coordinates) to s and t
                if 0 <= s < image.shape[0] and 0 <= t < image.shape[1]:     # check if the pixel is within the image boundaries   
                    max_value = max(max_value, image[s, t])                 # update the maximum valueby choosing the maximum of the current max value and the pixel value at s, t
            dilation_image[i, j] = max_value                                # update the current pixel value with the maximum value
    return dilation_image

# aim: (b) erosion
# explain: grayscale erosion assign each pixel in the image with the minimum value of the pixels covered by the kernel
def erosion(image, kernel):
    erosion_image  = np.zeros(image.shape, np.uint8)                       # initialize the erosion image with zeros of the same size as the original image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            min_value = 255
            for k in range(len(kernel)):                                    # iterate through the elements in the kernel
                s, t = kernel[k][0] + i, kernel[k][1] + j                   # assign (the coordinates of the current element in the kernel + the current pixel coordinates) to s and t
                if 0 <= s < image.shape[0] and 0 <= t < image.shape[1]:     # check if the pixel is within the image boundaries   
                    min_value = min(min_value, image[s, t])                 # update the minimum valueby choosing the minimum of the current min value and the pixel value at s, t
            erosion_image[i, j] = min_value                                 # update the current pixel value with the minimum value
    return erosion_image

# aim: (c) opening
# explain: the grayscale opening operation is the grayscale erosion followed by the grayscale dilation of the image with the same kernel
def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)

# aim: (d) closing
# explain: the grayscale closing operation is the grayscale dilation followed by the grayscale erosion of the image with the same kernel   
def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)


def main():
    image = cv2.imread('lena.bmp', 0)

    # aim: define the octogonal 3-5-5-5-3 kernel
    kernel = [
        [-2, -1], [-2, 0], [-2, 1],                     #3
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],  #5
        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],       #5
        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],       #5
        [2, -1], [2, 0], [2, 1]                         #3
    ]

    dilation_image = dilation(image, kernel)
    cv2.imshow('dilation_image', dilation_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    erosion_image = erosion(image, kernel)
    cv2.imshow('erosion_image', erosion_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    opening_image = opening(image, kernel)
    cv2.imshow('opening_image', opening_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    closing_image = closing(image, kernel)
    cv2.imshow('closing_image', closing_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()