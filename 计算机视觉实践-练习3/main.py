import cv2
import os

def upscale(img, alg_name, scale):
    # Create an SR object
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # Read the desired model
    path = f"./model/{alg_name}_x{scale}.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel(alg_name, scale)
    # Upscale the image
    result = sr.upsample(img)
    return result


def getQualityValues(upsampled, orig):
    psnr = cv2.PSNR(upsampled, orig)
    q, _ = cv2.quality.QualitySSIM_compute(upsampled, orig)
    ssim = (q[0] + q[1] + q[2]) / 3
    return round(psnr, 3), round(ssim, 3)


if __name__ == "__main__":
    # 图片路径
    img_path = "./Set5/baby.png"
    # 算法名称 edsr, espcn, fsrcnn or lapsrn
    algorithm = "lapsrn"
    # 放大系数
    scale = 4
    # 模型路径，根据算法确定
    model = f"./model/{algorithm}_x{scale}.pb"
    # 裁剪图像，使图像对齐
    img = cv2.imread(img_path)
    width = img.shape[0] - (img.shape[0] % scale)
    height = img.shape[1] - (img.shape[1] % scale)
    cropped = img[0:width, 0:height]
    # Downscale the image for benchmarking
    # 缩小图像，以实现基准质量测试
    img_downscaled = cv2.resize(cropped, None, fx=1.0 / scale, fy=1.0 / scale)
    img_new = upscale(img_downscaled, algorithm, scale)

    # 创建文件夹保存处理后的图像
    output_folder = f"./output/{algorithm}_x{scale}"
    os.makedirs(output_folder, exist_ok=True)



    # 获得模型质量评估值
    psnr, ssim = getQualityValues(cropped, img_new)
    print("LapSRN超分辨率算法和不同差值的性能对比")
    print("#" * 30)
    print(f"{algorithm}_x{scale}\nPSNT:{psnr}, SSIM:{ssim}")
    print("#" * 30)
    # INTER_CUBIC - 三次样条插值放大图像
    bicubic = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    psnr, ssim = getQualityValues(cropped, bicubic)
    print(f"bicubic插值\nPSNT:{psnr}, SSIM:{ssim}")
    print("#" * 30)
    # INTER_NEAREST - 最近邻插值
    nearest = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    psnr, ssim = getQualityValues(cropped, nearest)
    print(f"最近邻插值\nPSNT:{psnr}, SSIM:{ssim}")
    print("#" * 30)
    # Lanczos插值
    lanczos = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4);
    psnr, ssim = getQualityValues(cropped, lanczos)
    print(f"Lanczos插值\nPSNT:{psnr}, SSIM:{ssim}")
    print("#" * 30)


    # 保存超分辨率处理结果
    cv2.imwrite(f"{output_folder}/upscaled_image.png", img_new)

    # 保存其他插值方法的处理结果
    bicubic = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{output_folder}/bicubic.png", bicubic)

    nearest = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{output_folder}/nearest.png", nearest)

    lanczos = cv2.resize(img_downscaled, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f"{output_folder}/lanczos.png", lanczos)