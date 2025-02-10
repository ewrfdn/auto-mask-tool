#!/bin/bash

ENV_NAME="autocut"
PYTHON_VERSION="3.12"
ENV_PATH="./$ENV_NAME"

# Check architecture
ARCH=$(uname -m)
IS_ARM=false
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    IS_ARM=true
    echo "Detected ARM64 architecture"
else
    echo "Detected x86_64 architecture"
fi

# Check and install Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python not found, installing Python ${PYTHON_VERSION}..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS specific installation
            if [ "$IS_ARM" = true ]; then
                # ARM64 Mac
                arch -arm64 brew install python@${PYTHON_VERSION}
            else
                # Intel Mac
                brew install python@${PYTHON_VERSION}
            fi
        else
            # Linux installation
            if command -v apt &> /dev/null; then
                sudo apt update
                sudo apt install -y python${PYTHON_VERSION}
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y python${PYTHON_VERSION}
            else
                echo "Error: Unable to install Python automatically. Please install Python ${PYTHON_VERSION} manually."
                exit 1
            fi
        fi
    fi
}

# Check Python version and create venv
check_python

# Check if venv exists
if [ -d "$ENV_PATH" ]; then
    echo "Virtual environment exists, activating..."
else
    echo "Creating virtual environment '$ENV_NAME'..."
    python3 -m venv $ENV_PATH
fi

# Activate venv and update pip
source "$ENV_PATH/bin/activate"
python -m pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo -e "\nVirtual environment setup complete!"
echo -e "\nUsage:"
echo "1. Activate: source $ENV_PATH/bin/activate"
echo "2. Deactivate: deactivate"
echo "3. Environment path: $ENV_PATH"
