Jetson install
- python3 -m venv .env
- pip3 install --upgrade pip
- Instructions: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

- pip3 install opencv-python <- WRONG
- ln -s /usr/lib/python3.6/dist-packages/cv2 .env/lib/python3.6/site-packages/cv2 <- Correct way to bring the native python installation within the virtual environment
