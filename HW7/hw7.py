import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
length = img.shape[0]

img[img < 128] = 0
img[img >= 128] = 255

downsampled_img = np.zeros((64, 64))
for i in range(64):
    for j in range(64):
        downsampled_img[i][j] = img[i*8][j*8]

cv2.imwrite('downsampled_lena.bmp', downsampled_img)

def yokoi_connectivity_number(img):
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'
        return 's'

    def f(a1, a2, a3, a4):
        if a1 == a2 and a2 == a3 and a3 == a4 and a4 == 'r':
            return 5
        return [a1, a2, a3, a4].count('q')
    
    result = np.zeros((64, 64), dtype=int)
    for i in range(64):
        for j in range(64):
            if img[i][j] == 255:
                x0 = img[i][j]
                x1 = img[i][j+1] if j+1 < 64 else 0
                x2 = img[i-1][j] if i-1 >= 0 else 0
                x3 = img[i][j-1] if j-1 >= 0 else 0
                x4 = img[i+1][j] if i+1 < 64 else 0
                x5 = img[i+1][j+1] if i+1 < 64 and j+1 < 64 else 0
                x6 = img[i-1][j+1] if i-1 >= 0 and j+1 < 64 else 0
                x7 = img[i-1][j-1] if i-1 >= 0 and j-1 >= 0 else 0
                x8 = img[i+1][j-1] if i+1 < 64 and j-1 >= 0 else 0

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)
                result[i][j] = str(f(a1, a2, a3, a4))
            else:
                result[i][j] = 0
    return result

def pair_relation_operator(yokoi_result):
    def h(a, m):
        return int(a == m)
    
    def y(x0, x1, x2, x3, x4, m=1):
        sum = h(x1, m) + h(x2, m) + h(x3, m) + h(x4, m)
        return 'q' if sum < 1 or x0 != m \
                else 'p' if sum >= 1 and x0 == m \
                else ' '

    result = np.zeros((64, 64), dtype=str)
    for i in range(64):
        for j in range(64):
            if yokoi_result[i][j] == 1:
                x0 = yokoi_result[i][j]
                x1 = yokoi_result[i][j+1] if j+1 < 64 else 0
                x2 = yokoi_result[i-1][j] if i-1 >= 0 else 0
                x3 = yokoi_result[i][j-1] if j-1 >= 0 else 0
                x4 = yokoi_result[i+1][j] if i+1 < 64 else 0
                result[i][j] = y(x0, x1, x2, x3, x4)
            else:
                result[i][j] = ' '
    return result

def connect_shrink_operator(img, pair_result):
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 1
        return 0
    
    def f(a1, a2, a3, a4):
        if sum([a1, a2, a3, a4]) == 1:
            return 1
        return 0
    
    result = np.zeros((64, 64), dtype=str)
    for i in range(64):
        for j in range(64):
            if pair_result[i][j] == 'p':
                x0 = 255 if img[i][j] > 0 else 0
                x1 = 255 if (j+1 < 64 and img[i][j+1] > 0) else 0
                x2 = 255 if (i-1 >= 0 and img[i-1][j] > 0) else 0
                x3 = 255 if (j-1 >= 0 and img[i][j-1] > 0) else 0
                x4 = 255 if (i+1 < 64 and img[i+1][j] > 0) else 0
                x5 = 255 if (i+1 < 64 and j+1 < 64 and img[i+1][j+1] > 0) else 0
                x6 = 255 if (i-1 >= 0 and j+1 < 64 and img[i-1][j+1] > 0) else 0
                x7 = 255 if (i-1 >= 0 and j-1 >= 0 and img[i-1][j-1] > 0) else 0
                x8 = 255 if (i+1 < 64 and j-1 >= 0 and img[i+1][j-1] > 0) else 0

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)
                
                if f(a1, a2, a3, a4) == 1:
                    img[i][j] = 0
                    result[i][j] = 'g'
                else:
                    result[i][j] = 'p'
            else:
                result[i][j] = pair_result[i][j]
    
    return img, result

def thinning(img):
    iteration = 1
    prev_img = None
    current_img = img.copy()
    
    while not np.array_equal(prev_img, current_img):
        prev_img = current_img.copy()
        
        # Calculate Yokoi connectivity number
        yokoi_result = yokoi_connectivity_number(current_img)
        # np.savetxt(f'yokoi_connectivity_number_{iteration}.txt', yokoi_result, fmt='%d', delimiter=' ')
        
        # Apply pair relationship operator
        pair_result = pair_relation_operator(yokoi_result)
        # np.savetxt(f'pair_relation_operator_{iteration}.txt', pair_result, fmt='%s', delimiter=' ')
        
        # Apply connected shrink operator
        current_img, connect_result = connect_shrink_operator(current_img, pair_result)
        # np.savetxt(f'connect_shrink_operator_{iteration}.txt', connect_result, fmt='%s', delimiter=' ')
        cv2.imwrite(f'connect_shrink_lena_{iteration}.bmp', current_img)
        
        print(f"Iteration {iteration} completed")
        iteration += 1
    
    return current_img

def create_thinning_gif():
    import glob
    from PIL import Image
    
    filenames = glob.glob('connect_shrink_lena_*.bmp')
    filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    images = []
    images.append(Image.open('downsampled_lena.bmp'))
    for filename in filenames:
        img = Image.open(filename)
        images.append(img)
    
    duration = 200  # ms
    
    images[0].save(
        'thinning_process.gif',
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

final_img = thinning(downsampled_img)
cv2.imwrite('final_thinned_lena.bmp', final_img)
create_thinning_gif()