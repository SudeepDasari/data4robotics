#!/usr/bin/env bash

# make sure the folder doesn't already exist
if [ -d "visual_features" ]; then
    echo "Data already downloaded!"
    exit 0
fi

wget --output-document features.zip https://cmu.box.com/shared/static/rrlrp5g6ynk03io4rj9uzf6nik5urfl6
unzip features.zip
rm features.zip
