import sys
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
import numpy as np
import torch.nn.functional as F
import imageio.v3 as iio
from PIL import Image, ImageFilter, ImageEnhance
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# 移除全局的模型加载
model = None
device = None

def initialize_model():
    global model, device
    if model is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)
        model.to(device)

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

def postprocess_image(result: torch.Tensor, im_size: list, threshold=0.1, edge_hardness=0.95)-> np.ndarray:
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

def remove_background(input_path: str, output_path: str, model_size: list = [1024, 1024]) -> None:
    """
    移除图片背景
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param model_size: 模型输入尺寸
    """
    # 确保模型已加载
    initialize_model()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 使用 PIL 读取图片并确保是 RGB 格式
    with Image.open(input_path) as img:
        if (img.mode != 'RGB'):
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

def process_directory(input_dir: str, output_dir: str, model_size: list) -> None:
    """
    批量处理文件夹中的图片
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    :param model_size: 模型输入尺寸
    """
    # 确保模型已加载
    initialize_model()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    # 获取所有图片文件
    image_files = [
        f for f in Path(input_dir).rglob('*')
        if f.suffix.lower() in supported_formats
    ]
    
    if not image_files:
        print(f"在目录 {input_dir} 中没有找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    # 使用tqdm显示进度条
    for img_path in tqdm(image_files):
        # 构建输出路径，保持相对路径结构
        rel_path = img_path.relative_to(input_dir)
        output_path = Path(output_dir) / rel_path.with_suffix('.png')
        
        # 确保输出文件的父目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            remove_background(str(img_path), str(output_path), model_size)
        except Exception as e:
            print(f"\n处理失败 {img_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='AutoCut - AI图片背景移除工具\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  直接处理：
    autocut 图片.jpg                        # 输出到 图片_nobg.png
    
  单文件处理:
    autocut -i input.jpg                    # 输出到 input_nobg.png
    autocut -i input.jpg -s output.png      # 指定输出文件
    
  文件夹批量处理:
    autocut -d input_folder                 # 输出到 input_folder_nobg
    autocut -d input_folder -o output_dir   # 指定输出目录
    
  调整处理质量:
    autocut -i input.jpg --size 2048        # 更高质量（更慢）
    autocut -i input.jpg --size 512         # 更快处理（质量降低）
    
支持的图片格式: PNG, JPG, JPEG, WebP, BMP
'''
    )
    
    # 添加位置参数用于直接处理文件
    parser.add_argument('file', nargs='?', help=argparse.SUPPRESS)
    
    input_group = parser.add_argument_group('输入选项（可选其一）')
    input_group.add_argument('-i', '--input', 
                           help='单个图片的输入路径')
    input_group.add_argument('-d', '--dir', 
                           help='要处理的图片文件夹路径')
    
    output_group = parser.add_argument_group('输出选项（可选）')
    output_group.add_argument('-s', '--save',
                           help='输出图片保存路径（针对单个文件，默认为原文件名_nobg.png）')
    output_group.add_argument('-o', '--output',
                           help='输出文件夹路径（针对文件夹处理，默认为原文件夹名_nobg）')
    
    parser.add_argument('--size', type=int, default=1024,
                      help='模型处理尺寸，更大的值质量更好但更慢，更小的值处理更快但质量降低 (默认: 1024)')

    args = parser.parse_args()
    
    # 检查是否有任何参数
    if len(sys.argv) == 1:
        print("""
AutoCut - AI 智能抠图工具
------------------------
这是一个基于深度学习的图片背景去除工具，可以处理单张图片或批量处理文件夹。

基础用法：
  autocut 图片.jpg            # 直接处理单个文件
  autocut -h                  # 查看完整帮助信息

主要功能：
- 自动识别并移除图片背景
- 支持批量处理整个文件夹
- 支持多种图片格式
- 可调节处理质量
""")
        return
    
    # 如果是请求帮助信息，不加载模型
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        return

    # 处理直接输入的文件
    if args.file and not args.input and not args.dir:
        input_path = Path(args.file)
        if not input_path.is_file():
            print(f"错误：输入文件不存在: {input_path}")
            return
        output_path = input_path.with_name(f"{input_path.stem}_nobg.png")
        remove_background(str(input_path), str(output_path), [args.size, args.size])
        return

    # 处理单个文件
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"错误：输入文件不存在: {input_path}")
            return
            
        # 设置输出路径
        if args.save:
            output_path = Path(args.save)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_nobg.png")
            
        remove_background(str(input_path), str(output_path), [args.size, args.size])
        
    # 处理文件夹
    else:
        input_dir = Path(args.dir)
        if not input_dir.is_dir():
            print(f"错误：输入目录不存在: {input_dir}")
            return
            
        # 设置输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_dir.with_name(f"{input_dir.name}_nobg")
            
        process_directory(str(input_dir), str(output_dir), [args.size, args.size])

if __name__ == "__main__":
    main()