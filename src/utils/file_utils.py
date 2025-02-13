from pathlib import Path
from typing import Set, List, Optional

class PathManager:
    @staticmethod
    def ensure_output_directory(path: str | Path) -> Path:
        """确保输出目录存在，如果不存在则创建"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_relative_output_path(input_file: Path, 
                               input_dir: Path, 
                               output_dir: Path,
                               output_extension: str = '.png') -> Path:
        """获取输出文件的相对路径"""
        rel_path = input_file.relative_to(input_dir)
        return output_dir / rel_path.with_suffix(output_extension)

class FileScanner:
    DEFAULT_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    def __init__(self, 
                 allowed_formats: Optional[Set[str]] = None,
                 excluded_files: Optional[Set[str]] = None):
        """
        初始化文件扫描器
        :param allowed_formats: 允许的文件格式集合
        :param excluded_files: 要排除的具体文件名或路径
        """
        self.allowed_formats = allowed_formats or self.DEFAULT_IMAGE_FORMATS
        self.excluded_files = excluded_files or set()

    def should_include_file(self, file_path: Path) -> bool:
        """
        判断文件是否应该被包含
        :param file_path: 文件路径
        :return: 是否包含该文件
        """
        # 检查文件是否在排除列表中
        if str(file_path) in self.excluded_files or file_path.name in self.excluded_files:
            return False
        # 检查文件格式是否允许
        return file_path.suffix.lower() in self.allowed_formats

    def scan_directory(self, directory: str | Path, recursive: bool = True) -> List[Path]:
        """
        扫描目录获取所有符合要求的文件
        :param directory: 要扫描的目录
        :param recursive: 是否递归扫描子目录
        :return: 符合要求的文件路径列表
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"目录不存在: {directory}")

        pattern = "**/*" if recursive else "*"
        
        # 扫描并过滤文件
        files = [
            f for f in directory.glob(pattern)
            if f.is_file() and self.should_include_file(f)
        ]
        
        return sorted(files)
