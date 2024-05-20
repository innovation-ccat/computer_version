import os
import cv2
import numpy as np
from datetime import datetime

# 加载两幅图像
image1 = cv2.imread('./imgdata/img01.jpg')
image2 = cv2.imread('./imgdata/img02.jpg')

# 转换为灰度图
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 初始化ORB检测器
orb = cv2.ORB_create()

# 找到关键点和描述符
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# 初始化暴力匹配器
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配描述符
matches = matcher.match(descriptors1, descriptors2)

# 根据匹配的分数排序
matches = sorted(matches, key=lambda x: x.distance)

# 选择匹配分数最高的点对
matches_subset = matches[:10]

# 提取匹配点的坐标
points_src = np.zeros((len(matches_subset), 2))
points_dst = np.zeros((len(matches_subset), 2))

for i, match in enumerate(matches_subset):
    points_src[i] = keypoints1[match.queryIdx].pt
    points_dst[i] = keypoints2[match.trainIdx].pt

# 将点转换为np.float32类型，因为findHomography需要这个类型
points_src = np.float32(points_src)
points_dst = np.float32(points_dst)

# 使用RANSAC算法计算单应性变换矩阵
homography, _ = cv2.findHomography(points_src, points_dst, cv2.RANSAC)

# 打印单应性变换矩阵
print("单应性变换矩阵:\n", homography)

# 确保transform_result目录存在
output_dir = 'transform_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用单应性变换矩阵将原始图像映射到目标图像上
warped_image = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

# 获取当前时间戳，格式为 YYYYMMDD_HHMMSS_ffffff
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

# 生成唯一的文件名，例如：20230401_153045_123456.jpg
filename = f'transformed_image_{timestamp}.jpg'

# 保存路径
output_path = os.path.join(output_dir, filename)

# 显示原始图像和变换后的图像
cv2.imshow('Original Image', image1)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存变换后的图像
cv2.imwrite(output_path, warped_image)
print(f'Warped image saved as: {output_path}')