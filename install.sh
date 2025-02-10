#!/bin/bash

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

# Install package in editable mode
pip install -e .

echo "Installation complete! You can now use 'autocut' command anywhere."
echo "Example: autocut -i 1.jpg -o out.png"
