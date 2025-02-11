# AutoCut - AI 图片背景移除工具

一个使用 AI 技术快速移除图片背景的工具。

## 功能特点

- 自动识别并移除背景
- 支持多种图片格式
- 简单的命令行界面
- 跨平台支持（Windows/Linux/MacOS）

## 快速开始

1. 克隆此仓库
2. 运行初始化脚本：
   - Windows: `init.cmd`
   - Linux/MacOS: `./init.sh`
3. 运行工具：
   ```bash
   # Windows
   autocut -i 1.jpg -o out.png
   
   # Linux/MacOS
   ./autocut.sh -i 1.jpg -o out.png
   ```

## 环境要求

- Python 3.12+
- 其他依赖见 `requirements.txt`

## 使用方法
```bash
    # process file
    autocut -i image.jpg                    
    autocut -i image.jpg -s output.png      

    # process folder
    autocut -d input_folder                
    autocut -d input_folder -o output_dir   
```
## 安装说明

### Windows
```bash
# 克隆仓库
git clone https://github.com/yourusername/autocut.git
cd autocut

# 运行安装脚本
install.bat
```

### Linux/Mac
```bash
# 克隆仓库
git clone https://github.com/yourusername/autocut.git
cd autocut

# 运行安装脚本
chmod +x install.sh
./install.sh
```
