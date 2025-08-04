#!/bin/bash

# Install system dependencies
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

sudo apt update && sudo apt install bazel
sudo apt update && sudo apt install bazel-5.3.2

# Clone git repo to parent directory
git clone https://github.com/google/visqol ..

# Go to visqol directory where WORKSPACE file is located
cd ../visqol

# run bazel build

echo "Visqol installation completed successfully!"