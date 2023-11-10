#!/usr/bin/env bash

# Optional default path as the first argument
DEFAULT_PATH=$1

# Function to check if the directory exists
check_directory() {
    if [ -d "$1" ]; then
        echo "Data already downloaded in $1!"
        exit 0
    fi
}

# Check if the default path is provided and valid
if [ -n "$DEFAULT_PATH" ]; then
    # Check in the default path
    check_directory "${DEFAULT_PATH}/visual_features"
else
    # Check as-is
    check_directory "visual_features"
fi

# Download and extract process
wget --output-document features.zip https://cmu.box.com/shared/static/rrlrp5g6ynk03io4rj9uzf6nik5urfl6
unzip features.zip
rm features.zip