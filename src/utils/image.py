import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from PIL import Image, ImageFilter, ImageEnhance

def postprocess_image(result: torch.Tensor, im_size: list, threshold=0.95, edge_hardness=1.2)-> np.ndarray:
    """
    处理模型输出的掩码图像
    
    处理步骤：
    1. 尺寸调整和维度处理:
       - 使用双线性插值将结果调整到原始图像尺寸
       - 移除多余的维度
    
    2. 数值归一化:
       - 计算最大值和最小值
       - 将值缩放到0-1范围
    
    3. 阈值处理:
       - 低于threshold的值设为0
       - 高于edge_hardness的值设为1
       - 创建清晰的边界
    
    4. 格式转换:
       - 将值范围调整到0-255
       - 转换为numpy数组
       - 确保数据类型为uint8
    
    参数:
        result: 模型输出的掩码张量
        im_size: 目标图像尺寸
        threshold: 低阈值，低于此值视为背景
        edge_hardness: 高阈值，高于此值视为前景
    
    返回:
        处理后的掩码图像数组
    """
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    
    result = torch.where(result < threshold, torch.zeros_like(result), result)
    result = torch.where(result > edge_hardness, torch.ones_like(result), result)
    
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def enhance_mask(mask_image: Image.Image) -> Image.Image:
    """
    增强掩码图像质量
    
    处理步骤：
    1. 对比度增强:
       - 使用PIL的对比度增强器
       - 增强系数为1.5
    
    2. 边缘平滑:
       - 使用高斯模糊
       - 半径为0.5的轻微模糊
       - 减少锯齿状边缘
    
    参数:
        mask_image: 输入的掩码图像
    
    返回:
        增强后的掩码图像
    """
    enhancer = ImageEnhance.Contrast(mask_image)
    mask_image = enhancer.enhance(1.5)
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=0.5))
    return mask_image
