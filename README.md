# Correcting Gaze Using a Warping-Based Convolutional Neural Network

https://github.com/user-attachments/assets/8063acfe-7d8c-4ac1-a557-21bd94f6d6b4

## System Usage
To run the system, execute the following command:
```bash
python regz_socket_MP_FD.py
```

## Parameter Configuration
The parameters must be personalized in the `config.py` file before using the system. Their positions are illustrated in the following figure. The original point `P_o (0,0,0)` is defined at the center of the screen.

### Parameters to be personalized:
- `P_c_x`, `P_c_y`, `P_c_z`: Relative distance between the camera position and screen center (cm)
- `S_W`, `S_H`: Screen width and height (cm)
- `f`: Focal length of the camera

## Camera Focal Length Calibration
To estimate the focal length (`f`), execute one of the following scripts:
```bash
python focal_length_calibration.py
```
or open the Jupyter Notebook:
```bash
jupyter notebook focal_length_calibration.ipynb
```

### Steps for Calibration:
1. Position your head approximately 50 cm in front of the camera (modifiable in the code).
2. Insert your interpupillary distance (distance between the eyes) in the code, or use the average value of **6.3 cm**.
3. The estimated focal length will be displayed in the top-left corner of the window.

## Running Gaze Correction (Self-Demo)
- Press **'r'** while focusing on the "local" window and gaze at the "remote" window to start gaze correction.
- Press **'q'** while focusing on the "local" window to exit the program.

**Note:** The video may experience an initial delay due to TCP socket transmission, but will synchronize after a few seconds.

## Online Video Communication
Both local and remote systems use the same code, but the following parameters must be configured:
- `tar_ip`: The IP address of the other user
- `sender_port`: Port number for sending the redirected gaze video
- `recver_port`: Port number for receiving the redirected gaze video

## IP Setup for Self-Demo
For a self-demo, configure the following parameters:
- `tar_ip`: `127.0.0.1`
- `sender_port`: `5005`
- `recver_port`: `5005`

## Environmental Setup
Ensure your environment meets the following requirements:
- **Python**: 3.5.3
- **TensorFlow**: 1.8.0
- **CUDA**: V9.0.176 and corresponding cuDNN

## Required Packages
Install the following dependencies before running the system:
- Dlib 18.17.100
- OpenCV 3.4.1
- Numpy 1.15.4 + MKL
- pypiwin32
- Scipy 0.19.1

## DIRL Gaze Dataset
The dataset consists of gaze data collected from **37 Asian volunteers**:
- Approximately **100 gaze directions** per participant
- Horizontal range: **+40 to -40 degrees**
- Vertical range: **+30 to -30 degrees**
- **63 fixed** gaze directions, **37 random** gaze directions
- Images with closed eyes were removed

