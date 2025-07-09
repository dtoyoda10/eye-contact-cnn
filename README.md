# Gaze Correction via Warping-Based Convolutional Neural Network

## System Usage

Run the main script:

```bash
python regz_socket_MP_FD.py
```

---

## Focal Length Calibration

To estimate the focal length (`f`) of your camera:

1. Place your head \~50 cm from the camera (configurable).
2. Set your interpupillary distance (default: 6.3 cm).
3. Run one of the following:

```bash
python focal_length_calibration.py
# or
open focal_length_calibration.ipynb
```

The calculated focal length will appear at the top-left corner of the display.

---

## Start Gaze Correction (Self-Demo)

* Focus the **local window**, press **`r`** to start correction.
* Press **`q`** to quit.

> ⚠️ Initial delay may occur due to TCP socket transmission. Video will sync after a few seconds.

---

## Online Video Communication Setup

Use the same codebase for both **local** and **remote** sides.

Define the following in both configurations:

```python
tar_ip       = "<other_user_ip>"   # or "127.0.0.1" for self-demo
sender_port  = 5005
recver_port  = 5005
```

---

## Environment Requirements

* Python 3.5.3
* TensorFlow 1.8.0
* CUDA v9.0.176 with compatible cuDNN

---

## Required Packages

* `dlib==18.17.100`
* `opencv-python==3.4.1`
* `numpy==1.15.4` (with MKL)
* `pypiwin32`
* `scipy==0.19.1`
