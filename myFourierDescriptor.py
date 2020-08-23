import cv2
import numpy as np
import myCommonModules as cm


# 计算傅里叶描述子
# 输入：轮廓
# descLen=16，描述子长度
# 返回：描述子是否达到有效长度valid，傅里叶描述子descirptor的模
def fourierDesciptor(handContour,descLen=8):
    handContour=handContour[:,0,:]
    contours_complex = np.empty(handContour.shape[:-1], dtype=complex)
    contours_complex.real = handContour[:, 0]  # 横坐标作为实数部分
    contours_complex.imag = handContour[:, 1]  # 纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)  # 进行傅里叶变换
    # 是否达到有效长度
    valid = len(fourier_result) >= descLen
    # print(len(handContour),len(fourier_result),'>=', descLen,valid)
    descirptor = truncate_descriptor(fourier_result, descLen)  # 截短傅里叶描述子
    # reconstruct(np.zeros((480,640)), descirptor)
    descirptor_norm = abs(descirptor)  # 复数取模
    descirptor_norm = descirptor_norm/descirptor_norm[0] # 归一化
    # print(descirptor_norm)
    return valid, descirptor_norm

# 按maxLen截短傅里叶描述子
def truncate_descriptor(fourier_result, maxLen):
    descriptors_in_use = np.fft.fftshift(fourier_result)
    # 取中间的maxLen项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(maxLen / 2), center_index + int(maxLen / 2)
    descriptors_in_use = descriptors_in_use[low:high]
    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use


##由傅里叶描述子重建轮廓图，仅调试用
def reconstruct(img, descirptor_in_use):
    # descirptor_in_use = truncate_descriptor(fourier_result, degree)
    # descirptor_in_use = np.fft.ifftshift(fourier_result)
    # descirptor_in_use = truncate_descriptor(fourier_result)
    # print(descirptor_in_use)
    contour_reconstruct = np.fft.ifft(descirptor_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real,
                                    contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)

    black_np = np.ones(img.shape, np.uint8)  # 创建黑色幕布
    black = cv2.drawContours(black_np, contour_reconstruct, -1, (255, 255, 255), 10)  # 绘制白色轮廓
    cm.showImg(black,"contour_reconstruct",xScale=0.8)
    # cv2.imwrite('recover.png',black)
    return black
#