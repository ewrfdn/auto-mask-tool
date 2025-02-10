#!/bin/bash

ENV_NAME="autocut"
ENV_PATH="./$ENV_NAME"

# Check if virtual environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run './init.sh' first to set up the environment."
    exit 1
fi

# Activate virtual environment
source "$ENV_PATH/bin/activate"

# Run the main script
echo "Starting AutoCut..."
python ./src/autocut.py "$@"

# Deactivate virtual environment
deactivate
