import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# 讀取圖片
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise Exception("Could not read the image.")
length = img.shape[0]

# (a) Generate noisy images with gaussian noise (amplitude of 10 and 30)
def add_gaussian_noise(img, amplitude):
    noisy_img = img.copy()
    for i in range(length):
        for j in range(length):
            noise = random.gauss(0, amplitude)
            noisy_img[i][j] = np.clip(img[i][j] + noise, 0, 255)
    return noisy_img.astype(np.uint8)

# (b) Generate noisy images with salt-and-pepper noise (probability 0.1 and 0.05)
def add_salt_and_pepper_noise(img, probability):
    noisy_img = img.copy()
    for i in range(length):
        for j in range(length):
            if random.random() < probability:
                noisy_img[i][j] = 0
            elif random.random() < probability:
                noisy_img[i][j] = 255
    return noisy_img

# (c) Use 3x3, 5x5 box filter
def box_filter(img, size):
    result = np.zeros_like(img)
    half = size // 2
    
    for i in range(length):
        for j in range(length):
            total = 0
            count = 0
            # 處理 size x size 的窗口
            for x in range(-half, half + 1):
                for y in range(-half, half + 1):
                    # 檢查邊界
                    if (0 <= i + x < length) and (0 <= j + y < length):
                        total += img[i + x][j + y]
                        count += 1
            # 計算平均值
            result[i][j] = total // count
    return result

# (d) Use 3x3, 5x5 median filter
def median_filter(img, size):
    result = np.zeros_like(img)
    half = size // 2
    
    for i in range(length):
        for j in range(length):
            values = []
            # 處理 size x size 的窗口
            for x in range(-half, half + 1):
                for y in range(-half, half + 1):
                    # 檢查邊界
                    if (0 <= i + x < length) and (0 <= j + y < length):
                        values.append(img[i + x][j + y])
            # 計算中位數
            values.sort()
            result[i][j] = values[len(values) // 2]
    return result

# (e) Morphological operations
def dilation(img, kernel):
    m = len(kernel)
    n = len(kernel[0])
    mm = len(img)
    nn = len(img[0])
    result = np.zeros_like(img)  # 改用 np.zeros_like 確保資料型態一致
    
    for i in range(m//2, mm-m//2):
        for j in range(n//2, nn-n//2):
            max_val = 0
            for k in range(m):
                for l in range(n):
                    if kernel[k][l] == 1:
                        max_val = max(max_val, img[i-(m//2-k)][j-(n//2-l)])
            result[i][j] = max_val
    return result.astype(np.uint8)  # 確保輸出是 uint8 型態

def erosion(img, kernel):
    m = len(kernel)
    n = len(kernel[0])
    mm = len(img)
    nn = len(img[0])
    result = np.zeros_like(img)
    
    for i in range(m//2, mm-m//2):
        for j in range(n//2, nn-n//2):
            min_val = 255
            for k in range(m):
                for l in range(n):
                    if kernel[k][l] == 1:
                        min_val = min(min_val, img[i-(m//2-k)][j-(n//2-l)])
            result[i][j] = min_val
    return result.astype(np.uint8)

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# SNR 計算
def calculate_snr(original, processed):
    # 正規化到 0-1
    original = original.astype(float) / 255.0
    processed = processed.astype(float) / 255.0
    
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - processed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# 定義八角形 kernel
octagonal_kernel = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]
])

# 生成所有雜訊圖片
gaussian_10 = add_gaussian_noise(img, 10)
gaussian_30 = add_gaussian_noise(img, 30)
sp_01 = add_salt_and_pepper_noise(img, 0.1)
sp_005 = add_salt_and_pepper_noise(img, 0.05)

# 儲存所有處理結果和對應的 SNR
results = {
    'Gaussian (σ=10)': {
        'Noisy': (gaussian_10, calculate_snr(img, gaussian_10)),
        '3x3 Box': (box_filter(gaussian_10, 3), calculate_snr(img, box_filter(gaussian_10, 3))),
        '5x5 Box': (box_filter(gaussian_10, 5), calculate_snr(img, box_filter(gaussian_10, 5))),
        '3x3 Median': (median_filter(gaussian_10, 3), calculate_snr(img, median_filter(gaussian_10, 3))),
        '5x5 Median': (median_filter(gaussian_10, 5), calculate_snr(img, median_filter(gaussian_10, 5))),
        'Opening-Closing': (closing(opening(gaussian_10, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, closing(opening(gaussian_10, octagonal_kernel), octagonal_kernel))),
        'Closing-Opening': (opening(closing(gaussian_10, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, opening(closing(gaussian_10, octagonal_kernel), octagonal_kernel)))
    },
    'Gaussian (σ=30)': {
        'Noisy': (gaussian_30, calculate_snr(img, gaussian_30)),
        '3x3 Box': (box_filter(gaussian_30, 3), calculate_snr(img, box_filter(gaussian_30, 3))),
        '5x5 Box': (box_filter(gaussian_30, 5), calculate_snr(img, box_filter(gaussian_30, 5))),
        '3x3 Median': (median_filter(gaussian_30, 3), calculate_snr(img, median_filter(gaussian_30, 3))),
        '5x5 Median': (median_filter(gaussian_30, 5), calculate_snr(img, median_filter(gaussian_30, 5))),
        'Opening-Closing': (closing(opening(gaussian_30, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, closing(opening(gaussian_30, octagonal_kernel), octagonal_kernel))),
        'Closing-Opening': (opening(closing(gaussian_30, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, opening(closing(gaussian_30, octagonal_kernel), octagonal_kernel)))
    },
    'S&P (p=0.1)': {
        'Noisy': (sp_01, calculate_snr(img, sp_01)),
        '3x3 Box': (box_filter(sp_01, 3), calculate_snr(img, box_filter(sp_01, 3))),
        '5x5 Box': (box_filter(sp_01, 5), calculate_snr(img, box_filter(sp_01, 5))),
        '3x3 Median': (median_filter(sp_01, 3), calculate_snr(img, median_filter(sp_01, 3))),
        '5x5 Median': (median_filter(sp_01, 5), calculate_snr(img, median_filter(sp_01, 5))),
        'Opening-Closing': (closing(opening(sp_01, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, closing(opening(sp_01, octagonal_kernel), octagonal_kernel))),
        'Closing-Opening': (opening(closing(sp_01, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, opening(closing(sp_01, octagonal_kernel), octagonal_kernel)))
    },
    'S&P (p=0.05)': {
        'Noisy': (sp_005, calculate_snr(img, sp_005)),
        '3x3 Box': (box_filter(sp_005, 3), calculate_snr(img, box_filter(sp_005, 3))),
        '5x5 Box': (box_filter(sp_005, 5), calculate_snr(img, box_filter(sp_005, 5))),
        '3x3 Median': (median_filter(sp_005, 3), calculate_snr(img, median_filter(sp_005, 3))),
        '5x5 Median': (median_filter(sp_005, 5), calculate_snr(img, median_filter(sp_005, 5))),
        'Opening-Closing': (closing(opening(sp_005, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, closing(opening(sp_005, octagonal_kernel), octagonal_kernel))),
        'Closing-Opening': (opening(closing(sp_005, octagonal_kernel), octagonal_kernel),
                          calculate_snr(img, opening(closing(sp_005, octagonal_kernel), octagonal_kernel)))
    }
}

# 顯示結果時調整版面
# 顯示結果
plt.figure(figsize=(30, 24))  # 加大整體尺寸

# 先創建外部的大標題
plt.suptitle('Image Processing Results', fontsize=12, y=0.95)

# 為每種噪聲類型創建子圖
for noise_name, noise_results in results.items():
    plt.figure(figsize=(20, 3))  # 每種雜訊類型一個橫排
    plt.suptitle(f'{noise_name}', fontsize=12, y=1.05)
    
    for j, (process_name, (processed_img, snr)) in enumerate(noise_results.items()):
        plt.subplot(1, 7, j + 1)  # 1行7列
        plt.imshow(processed_img, cmap='gray')
        plt.title(f'{process_name}\nSNR: {snr:.2f}dB', fontsize=10)
        plt.axis('off')
    
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'result_{noise_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 調整子圖之間的間距
plt.subplots_adjust(
    left=0.05,    # 左邊界
    right=0.95,   # 右邊界
    top=0.92,     # 上邊界
    bottom=0.05,  # 下邊界
    wspace=0.35,  # 水平間距
    hspace=0.45   # 垂直間距
)

# 儲存圖片時使用更高的DPI以確保清晰度
plt.savefig('result.png', 
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.5)

plt.show()

# 打印 SNR 值表格 (保持不變)
print("\nSNR Values (dB):")
print("-" * 120)
header = f"{'Noise Type':<15} | {'Noisy':<10} | {'3x3 Box':<10} | {'5x5 Box':<10} | {'3x3 Median':<10} | {'5x5 Median':<10} | {'Open-Close':<10} | {'Close-Open':<10}"
print(header)
print("-" * 120)

for noise_name, noise_results in results.items():
    values = [f"{v[1]:.2f}" for v in noise_results.values()]
    print(f"{noise_name:<15} | {values[0]:<10} | {values[1]:<10} | {values[2]:<10} | {values[3]:<10} | {values[4]:<10} | {values[5]:<10} | {values[6]:<10}")