import torch
import numpy as np
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
import warnings
from pathlib import Path

# 禁用PIL的最大图片尺寸限制
Image.MAX_IMAGE_PIXELS = None

class BackgroundRemover:
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        
    def initialize_model(self):
        """初始化RMBG-2.0模型和预处理转换器"""
        if self.model is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 设置模型精度
            if self.device.type == "cuda":
                torch.set_float32_matmul_precision('high')
            
            # 初始化模型
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0",
                trust_remote_code=True,
                cache_dir="models"
            )
            self.model.to(self.device)
            self.model.eval()
            
            # 初始化转换器
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),  # 默认尺寸
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def remove_background(self, input_path: str, output_path: str, size: int = 2048) -> None:
        """移除图片背景"""
        self.initialize_model()
        
        try:
            # 读取图片
            image = Image.open(input_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # 更新transform的尺寸
            self.transform.transforms[0] = transforms.Resize((size, size))
            
            # 预处理和推理
            with torch.no_grad():
                # 预处理
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # 模型推理
                predictions = self.model(input_tensor)[-1]
                mask = predictions.sigmoid().cpu()[0].squeeze()
                
                # 将掩码转换为PIL图像并调整大小
                mask_pil = transforms.ToPILImage()(mask)
                mask_pil = mask_pil.resize(image.size)
                
                # 应用掩码
                result_image = image.copy()
                result_image.putalpha(mask_pil)
                
                # 保存结果
                output_path = Path(output_path).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result_image.save(str(output_path), format='PNG')
                    
        except Exception as e:
            raise RuntimeError(f"处理失败: {str(e)}")
