# AutoCut - AI Image Background Remover

A simple tool to remove image backgrounds using AI.

## Features

- Automatic background removal
- Support for multiple image formats
- Easy to use command line interface
- Cross-platform support (Windows/Linux/MacOS)

## Installation

### Windows
```bash
# Clone the repository
git clone https://github.com/yourusername/autocut.git
cd autocut

# Run the installation script
install.bat
```

### Linux/Mac
```bash
# Clone the repository
git clone https://github.com/yourusername/autocut.git
cd autocut

# Run the installation script
chmod +x install.sh
./install.sh
```

## Quick Start

1. Clone this repository
2. Run initialization script:
   - Windows: `init.cmd`
   - Linux/MacOS: `./init.sh`
3. Run the tool:
   ```bash
   # process file
    autocut -i image.jpg                    
    autocut -i image.jpg -s output.png      

    # process folder
    autocut -d input_folder                
    autocut -d input_folder -o output_dir   
   
   ```

## Requirements

- Python 3.12+
- See `requirements.txt` for Python dependencies

## Usage
