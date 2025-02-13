import sys
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from model.rmbg import BackgroundRemover
from utils.file_utils import FileScanner, PathManager

def process_directory(remover: BackgroundRemover, input_dir: str, output_dir: str, model_size: int) -> None:
    # 初始化文件扫描器和路径管理器
    file_scanner = FileScanner()
    
    try:
        # 获取所有图片文件
        image_files = file_scanner.scan_directory(input_dir)
        
        if not image_files:
            print(f"在目录 {input_dir} 中没有找到支持的图片文件")
            return
        
        print(f"找到 {len(image_files)} 个图片文件，开始处理...")
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        
        for img_path in tqdm(image_files):
            output_path = PathManager.get_relative_output_path(img_path, input_dir_path, output_dir_path)
            PathManager.ensure_output_directory(output_path.parent)
            
            try:
                remover.remove_background(str(img_path), str(output_path), model_size)
            except Exception as e:
                print(f"\n处理失败 {img_path}: {str(e)}")
                
    except Exception as e:
        print(f"处理目录时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='AutoCut - AI图片背景移除工具\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''示例用法:
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
                      help='模型处理尺寸，更大的值质量更好但更慢，更小的值处理更快但质量降低 (默认: 2048)')

    args = parser.parse_args()
    
    # 检查是否有任何参数
    if len(sys.argv) == 1:
        print("""AutoCut - AI 智能抠图工具
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

    remover = BackgroundRemover()

    # 处理直接输入的文件
    if args.file and not args.input and not args.dir:
        input_path = Path(args.file)
        if not input_path.is_file():
            print(f"错误：输入文件不存在: {input_path}")
            return
        output_path = input_path.with_name(f"{input_path.stem}_nobg.png")
        remover.remove_background(str(input_path), str(output_path), args.size)
        return

    # 处理单个文件
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"错误：输入文件不存在: {input_path}")
            return
            
        output_path = Path(args.save) if args.save else input_path.with_name(f"{input_path.stem}_nobg.png")
        remover.remove_background(str(input_path), str(output_path), args.size)
        
    # 处理文件夹
    elif args.dir:
        input_dir = Path(args.dir)
        if not input_dir.is_dir():
            print(f"错误：输入目录不存在: {input_dir}")
            return
            
        output_dir = Path(args.output) if args.output else input_dir.with_name(f"{input_dir.name}_nobg")
        process_directory(remover, str(input_dir), str(output_dir), args.size)

if __name__ == "__main__":
    main()