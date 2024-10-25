import cv2
import numpy as np


# aim: (a) Dilation
def dilation(bin_img, kernel):
    height = bin_img.shape[0]
    width = bin_img.shape[1]
    dilation_img = np.zeros(bin_img.shape, dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            # explain: here we consider 2 cases,
            # explain: first check that if a pixel is 255 or not, if it's 0, then we just skip for this pixel
            # explain: otherwise, we set the pixel in the dilation_img to be 255,
            # explain: at the same time, we add the translation of the elemtents in the kernel, and if the resulting pixels are within the image, we update them to be 255
            # note: here we ignore checking the value of the resulting pixels, but just set them all to be 255, no matter originally they are 0 or 255
            if bin_img[i][j] == 255:
                dilation_img[i][j] = 255
                for x, y in kernel:
                    ni, nj = i + x, j + y                       # here ni, nj represents the coordinate after translation
                    if 0 <= ni < height and 0 <= nj < width:    # check if the translated coordinate is within the image
                        dilation_img[ni][nj] = 255
    return dilation_img

# aim: (b) erosion
def erosion(bin_img, kernel):
    height = bin_img.shape[0]
    width = bin_img.shape[1]
    erosion_img = np.zeros(bin_img.shape, dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            # explain: we implement erosion by checking for an arbitrary pixel in the image, 
            # explain: if we translate the kernel to this pixel, and all the pixels within the kernel are 255, then we set the pixel in the erosion_img to be 255, otherwise 0
            # explain: which means that only set a pixel to be 255 if there's a fit
            for x, y in kernel:
                ni, nj = i + x, j + y
                if 0 <= ni < height and 0 <= nj < width:
                    if bin_img[ni][nj] == 0:
                        erosion_img[i][j] = 0
                        break
            else:
                erosion_img[i][j] = 255
    return erosion_img

# aim: (e) hit-and-miss transform
def hit_and_miss(bin_img, J_kernel, K_kernel):
    # explain: we implement hit_and_miss by definition, which means calling the erosion function twice, 
    # explain: first we do the erosion by J_kernel, and then do the erosion of the complement of the original image by K_kernel
    # explain: the final result would be the intersection of the two erosion results

    A_ero_Jkernel = erosion(bin_img, J_kernel)
    A_comp = 255 - bin_img
    Acomp_ero_Kkernel = erosion(A_comp, K_kernel)
    hit_and_miss_img = A_ero_Jkernel & Acomp_ero_Kkernel

    return hit_and_miss_img


def main():

    # aim: import the image and convert it to grayscale

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


    # aim: construct the kernels (where "kernel: is used for part (a) to (d), and "J_kernel", "K_kernel" is used for part (e))
    # subaim: the octogonal 3-5-5-5-3 kernel
    # explain: each element in the kernel list below is the coordinate with respect to the origin (0, 0)
    kernel = [[-2, -1], [-2, 0], [-2, 1],                       # 3
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],    # 5
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],         # 5
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],         # 5
              [2, -1], [2, 0], [2, 1]]                          # 3
    
    # subaim: the J_kernel and K_kernel for hit-and-miss transform
    J_kernel = [[0, -1], [0, 0], [1, 0]]
    K_kernel = [[-1, 0], [-1, 1], [0, 1]]
    

    # aim: do the (a) to (e) required operations and save the results
    # subaim: (a) dilation
    dilation_lena = dilation(bin_lena, kernel)
    cv2.imwrite('dilation_lena.bmp', dilation_lena)

    # subaim: (b) erosion
    erosion_lena = erosion(bin_lena, kernel)
    cv2.imwrite('erosion_lena.bmp', erosion_lena)

    # note: in the next 2 parts (c) and (d), we follow the definition of opening and closing, and implement them by calling the dilation and erosion function directly

    # subaim: (c) opening
    # explain: opening = erosion followed by dilation
    opening_lena = dilation(erosion(bin_lena, kernel), kernel)
    cv2.imwrite('opening_lena.bmp', opening_lena)

    # subaim: (d) closing
    # explain: closing = dilation followed by erosion
    closing_lena = erosion(dilation(bin_lena, kernel), kernel)
    cv2.imwrite('closing_lena.bmp', closing_lena)

    # subaim: (e) hit-and-miss transform
    hit_and_miss_lena = hit_and_miss(bin_lena, J_kernel, K_kernel)
    cv2.imwrite('hit_and_miss_lena.bmp', hit_and_miss_lena)
    

if __name__ == '__main__':
    main()
                