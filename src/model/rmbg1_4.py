import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from PIL import Image
from utils.image import postprocess_image, enhance_mask
import warnings
from pathlib import Path

# 禁用PIL的最大图片尺寸限制
Image.MAX_IMAGE_PIXELS = None
# 或者设置一个更大的值
# Image.MAX_IMAGE_PIXELS = 933120000  # 例如：30000x30000像素

def prepare_rmbg_input(image: np.ndarray, target_size: list) -> torch.Tensor:
    """
    为RMBG模型准备输入数据
    
    处理步骤：
    1. 通道检查和转换:
       - 确保图像是3通道(RGB)格式
       - 如果是单通道图像，将其扩展为3通道
    
    2. 转换为PyTorch张量:
       - 将numpy数组转换为PyTorch张量
       - 使用permute(2,0,1)调整维度顺序：
         * 从(H,W,C)转换为(C,H,W)
         * H=高度，W=宽度，C=通道数
    
    3. 调整图像尺寸:
       - 使用F.interpolate进行图像重采样
       - 添加batch维度：(C,H,W) -> (1,C,H,W)
       - 使用双线性插值调整到目标尺寸
    
    4. 像素值归一化:
       - 将像素值从0-255范围转换到0-1范围
       - 除以255.0进行归一化
    
    5. 标准化处理:
       - 使用normalize函数进行标准化
       - 均值[0.5,0.5,0.5]和标准差[1.0,1.0,1.0]
       - 将数据分布的中心移到0附近
       - 确保与模型训练时的数据分布一致
    
    参数:
        image: 输入的numpy数组格式图像 (H x W x C)
        target_size: RMBG模型需要的输入尺寸 [height, width]
    
    返回:
        预处理后的图像tensor，格式为(1,3,H,W)，值范围已标准化
    """
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
    
    im_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor,0),
        size=target_size,
        mode='bilinear'
    )
    
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0])
    
    return image

class BackgroundRemover:
    def __init__(self):
        self.model = None
        self.device = None
        
    def initialize_model(self):
        """初始化RMBG-2.0模型"""
        if self.model is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # 更新到RMBG-2.0版本
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0",
                trust_remote_code=True,
                cache_dir="models"  # 缓存模型到本地目录
            )
            self.model.to(self.device)
            # 设置为评估模式
            self.model.eval()

    def remove_background(self, input_path: str, output_path: str, size: int = 2048) -> None:
        """
        移除图片背景
        注意：RMBG-2.0支持更大的输入尺寸，默认改为2048以获得更好的效果
        """
        self.initialize_model()

        # 读取并预处理图片
        with Image.open(input_path) as img:
            if (img.mode != 'RGB'):
                img = img.convert('RGB')
            orig_im = np.array(img)

        orig_im_size = orig_im.shape[0:2]
        # 保持宽高比的同时调整大小
        aspect_ratio = orig_im_size[1] / orig_im_size[0]
        if aspect_ratio > 1:
            model_size = [size, int(size * aspect_ratio)]
        else:
            model_size = [int(size / aspect_ratio), size]

        # 使用torch.no_grad()优化推理性能
        with torch.no_grad():
            image = prepare_rmbg_input(orig_im, model_size).to(self.device)
            # 模型推理
            result = self.model(image)

        # 后处理
        result_image = postprocess_image(result[0][0], orig_im_size)
        pil_mask_im = Image.fromarray(result_image)
        pil_mask_im = enhance_mask(pil_mask_im)

        # 应用蒙版
        orig_image = Image.open(input_path)
        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask_im)

        # 保存结果
        try:
            # 转换为绝对路径并确保输出目录存在
            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 安全地保存文件
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # 忽略解压缩炸弹警告
                no_bg_image.save(
                    str(output_path),
                    format='PNG',
                    quality=95,
                    optimize=True
                )
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"保存文件失败 {output_path}: {str(e)}")
