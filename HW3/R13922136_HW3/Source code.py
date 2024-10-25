import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # aim: (a) generate original image and its histogram

    # explain: the flag 0 represents cv2.IMREAD_GRAYSCALE, which reads the image in grayscale mode
    orig_lena = cv2.imread('lena.bmp', 0)
    
    cv2.imshow('Original Lena', orig_lena)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist_orig_lena = np.zeros(256, dtype = np.uint32)
    for i in range(orig_lena.shape[0]):
        for j in range(orig_lena.shape[1]):
            hist_orig_lena[orig_lena[i][j]] += 1
    
    # subaim: plot the histogram of the original lena image
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of the Original Grayscale Image')
    plt.bar(np.arange(256), hist_orig_lena)
    plt.show()

    # aim: (b) image with intensity divided by 3 and its histogram
    lena_divided_by_3 = orig_lena // 3
    cv2.imshow('Lena divided by 3', lena_divided_by_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist_lena_divided_by_3 = np.zeros(256, dtype = np.uint32)
    for i in range(lena_divided_by_3.shape[0]):
        for j in range(lena_divided_by_3.shape[1]):
            hist_lena_divided_by_3[lena_divided_by_3[i][j]] += 1
    
    # subaim: plot the histogram of the lena image with intensity divided by 3
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of the Lena Image with Intensity Divided by 3')
    plt.bar(np.arange(256), hist_lena_divided_by_3)
    plt.show()

    # aim: (c) image after applying histogram equalization to (b) and its histogra
    
    # subaim: calculate cumulative distribution function (CDF)
    # explain: since hist_lena_divided_by_3 contains the pixel intensity counts, 
    # explain: we use np.cumsum to get the cumulative sum of the elements
    cdf = hist_lena_divided_by_3.cumsum()

    # subaim: normalize cdf
    # explain: we scale the cdf to the full range of intensity values (0 to 255), 
    # explain: and divide by the total number of pixels(which is the last element of cdf)
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # subaim: create lookup table
    lookup_table = np.uint8(cdf_normalized)
    
    # subaim: apply histogram equalization
    # explain: we use the lookup table to map the original pixel intensity values to the new intensity values
    # explain: the new intensity values are the indices of the cdf_normalized array
    # explain: the resulting array, lena_divided_by_3_equalized, will have its intensity values 
    # explain: uniformly distributed over the range of 0 to 255
    
    lena_divided_by_3_equalized = lookup_table[lena_divided_by_3]
    
    cv2.imshow('Lena divided by 3 equalized', lena_divided_by_3_equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist_lena_divided_by_3_equalized = np.zeros(256, dtype = np.uint32)
    for i in range(lena_divided_by_3_equalized.shape[0]):
        for j in range(lena_divided_by_3_equalized.shape[1]):
            hist_lena_divided_by_3_equalized[lena_divided_by_3_equalized[i][j]] += 1

    # subaim: plot the histogram of the lena image with intensity divided by 3 after histogram equalization
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of the Lena Image with Intensity Divided by 3 After Histogram Equalization')
    plt.bar(np.arange(256), hist_lena_divided_by_3_equalized)
    plt.show()

if __name__ == '__main__':
    main()