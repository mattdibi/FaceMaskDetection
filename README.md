## Jetson dependencies installation instructions

#### Create a new Python virtual environment

```
cd /path/to/this/repository && python3 -m venv .env
```

Then activate it:

```
source .env/bin/activate
```

#### Upgrade pip

```
pip3 install --upgrade pip
```

Tested version: 21.1.1

#### Install Tensorflow

Instructions available from the [Nvidia documentation](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)

```
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```

```
pip3 install -U pip testresources setuptools==49.6.0 
```

```
pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
```

```
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow
```

#### Install OpenCV

The Jetson already ship with an Nvidia-customized Opencv installation by default, we only need to bring this version available inside the virtual environment

```
ln -s /usr/lib/python3.6/dist-packages/cv2 .env/lib/python3.6/site-packages/cv2
```

## Running instructions

```
usage: main.py [-h] [--camera CAMERA] [--frame_interval FRAME_INTERVAL]
               [--url URL] [--auth_username AUTH_USERNAME]
               [--auth_password AUTH_PASSWORD]

Face Mask Detection

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA       Set input camera device
  --frame_interval FRAME_INTERVAL
                        Set the frame interval for post data request
  --url URL             Url of the RESTful API target for the post data
                        request
  --auth_username AUTH_USERNAME
                        Username for Basic Access Authentication required by
                        the RESTful API
  --auth_password AUTH_PASSWORD
                        Password for Basic Access Authentication required by
                        the RESTful API

```

Example:

```
python3 main.py --url "https://127.0.0.1/app/v1/core" --auth_username "admin" --auth_password "password"
```
