from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
import numpy as np
import torch.nn.functional as F
import imageio.v3 as iio
from PIL import Image, ImageFilter, ImageEnhance
import argparse
import os

model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)

# 可以调整这个值，但要注意：
model_input_size = [1024, 1024]  # 默认值
# 更大的值可能提供更好的细节，但会消耗更多内存和处理时间
# 更小的值处理更快，但可能丢失细节

# 如果想要更高质量，可以尝试：
# model_input_size = [1024,1024]  # 更高质量，但更慢
# 如果想要更快的处理：
# model_input_size = [256,256]    # 更快，但质量下降

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list, threshold=0.3, edge_hardness=0.95)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    
    # 强化阈值处理
    result = torch.where(result < threshold, torch.zeros_like(result), result)
    result = torch.where(result > edge_hardness, torch.ones_like(result), result)
    
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def remove_background(input_path: str, output_path: str, model_size: list = [1024, 1024]) -> None:
    """
    移除图片背景
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param model_size: 模型输入尺寸
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 使用 PIL 读取图片并确保是 RGB 格式
    with Image.open(input_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 将 PIL Image 转换为 numpy array
        orig_im = np.array(img)

    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_size).to(device)

    # inference 
    result = model(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_mask_im = Image.fromarray(result_image)

    # 增强对比度
    enhancer = ImageEnhance.Contrast(pil_mask_im)
    pil_mask_im = enhancer.enhance(1.5)

    # 轻微模糊处理边缘
    pil_mask_im = pil_mask_im.filter(ImageFilter.GaussianBlur(radius=0.5))

    orig_image = Image.open(input_path)
    no_bg_image = orig_image.copy()
    no_bg_image.putalpha(pil_mask_im)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 保存结果
    no_bg_image.save(output_path, quality=95, optimize=True)
    print(f"处理完成！输出文件：{output_path}")

def main():
    parser = argparse.ArgumentParser(description='图片背景移除工具')
    parser.add_argument('-i', '--input', required=True, help='输入图片路径')
    parser.add_argument('-o', '--output', help='输出图片路径')
    parser.add_argument('--size', type=int, default=1024, help='模型处理尺寸 (默认: 1024)')
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，则在输入文件名后添加_nobg
    if not args.output:
        input_name = os.path.splitext(args.input)[0]
        args.output = f"{input_name}_nobg.png"
    
    try:
        remove_background(args.input, args.output, [args.size, args.size])
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()