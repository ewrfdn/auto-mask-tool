#!/bin/bash
ENV_PATH="$AUTOCUT_ENV_PATH"
# Check if AUTOCUT_ENV_PATH environment variable exists
if [ -n "$AUTOCUT_ENV_PATH" ]; then
else
    ENV_NAME="autocut"
    ENV_PATH="./$ENV_NAME"
fi

# Check if virtual environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Virtual environment not found at $ENV_PATH!"
    echo "Please run './init.sh' first to set up the environment."
    exit 1
fi

# Activate virtual environment
source "$ENV_PATH/bin/activate"

# Run the main script
echo "Starting AutoCut..."
python "${ENV_PATH}/../src/autocut.py" "$@"

# Deactivate virtual environment
deactivate
