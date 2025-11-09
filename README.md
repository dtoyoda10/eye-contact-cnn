# Eye Contact CNN

A deep learning-based system for real-time gaze redirection in video streams. This project implements a CNN model that can manipulate eye gaze direction to create natural eye contact during video calls.

## Overview

Eye Contact CNN uses convolutional neural networks with optical flow warping to redirect eye gaze in real-time video streams. The system detects facial landmarks, analyzes eye position and gaze direction, and applies neural network-based transformations to create the appearance of direct eye contact.

## Features

- Real-time gaze redirection using CNN-based warping
- Facial landmark detection with dlib
- Socket-based video communication for video call applications
- Support for both left and right eye processing
- Configurable camera and screen parameters
- Cross-platform support (Windows, Linux, macOS)

## System Requirements

### Hardware
- CPU: Multi-core processor (Intel i5 or equivalent recommended)
- RAM: Minimum 8GB (16GB recommended)
- GPU: NVIDIA GPU with CUDA support (optional but recommended for better performance)
- Webcam: Any standard webcam (minimum 640x480 resolution)

### Software
- Python 3.6 or 3.7
- TensorFlow 1.14 or higher (TensorFlow 1.x only)
- OpenCV 4.2 or higher
- dlib 19.19 or higher

Note: This project currently uses TensorFlow 1.x APIs. Migration to TensorFlow 2.x would require code updates.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/eye-contact-cnn.git
cd eye-contact-cnn
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required models

Download the dlib facial landmark predictor:
```bash
# Download shape_predictor_68_face_landmarks.dat
# Place it in: gaze_correction_system/lm_feat/
```

You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place the .dat file in the correct location.

### 5. Download or train model weights

Place your trained model weights in:
```
gaze_correction_system/weights/warping_model/flx/12/
```

## Usage

### Basic Usage

Run the gaze correction system:

```bash
cd gaze_correction_system
python regz_socket_MP_FD.py
```

### Configuration

You can customize the system behavior using command-line arguments:

```bash
python regz_socket_MP_FD.py --height 48 --width 64 --ef_dim 12
```

Available configuration options:

- `--height`: Input image height (default: 48)
- `--width`: Input image width (default: 64)
- `--channel`: Number of color channels (default: 3)
- `--ef_dim`: Eye feature dimension (default: 12)
- `--tar_ip`: Target IP address (default: localhost)
- `--sender_port`: Sender socket port (default: 5005)
- `--recver_port`: Receiver socket port (default: 5005)
- `--f`: Focal length (default: 650)

### Training Your Own Model

To train a new model:

1. Prepare your dataset in the required format
2. Navigate to the training directory:

```bash
cd training/code_tf/model_train
```

3. Run the training script:

```bash
python model_train.py --dataset your_dataset_name --epochs 500
```

Training configuration options:

- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 500)
- `--batch_size`: Batch size (default: 256)
- `--tar_model`: Target model architecture (default: flx)

## Project Structure

```
eye-contact-cnn/
├── gaze_correction_system/      # Main inference/deployment code
│   ├── config.py                 # Configuration parameters
│   ├── flx.py                    # FLX model architecture
│   ├── regz_socket_MP_FD.py     # Main gaze correction system
│   ├── transformation.py         # Image transformation utilities
│   └── tf_utils.py              # TensorFlow utility functions
├── training/                     # Training pipeline
│   ├── code_tf/
│   │   └── model_train/         # Model training code
│   │       ├── config.py
│   │       ├── flx.py
│   │       └── model_train.py
│   └── data_generation/         # Dataset generation utilities
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Architecture

The system uses a multi-step pipeline:

1. **Face Detection**: Detects faces in the input video stream
2. **Landmark Detection**: Identifies 68 facial landmarks using dlib
3. **Eye Region Extraction**: Extracts eye regions based on landmarks
4. **Gaze Analysis**: Analyzes current gaze direction
5. **Neural Warping**: Applies CNN-based warping to redirect gaze
6. **Composition**: Blends the warped eyes back into the original frame

The neural network uses the FLX (Flexible) architecture with optical flow-based warping.

## Performance Considerations

- The system processes video in real-time on most modern hardware
- GPU acceleration significantly improves performance
- Network latency may affect socket-based communication
- Processing speed depends on input resolution and model complexity

## Known Limitations

- Currently optimized for single-person scenarios
- Performance may vary with different lighting conditions
- Requires clear frontal or near-frontal face views
- May struggle with extreme head poses or partial occlusions

## Troubleshooting

### Common Issues

**Issue**: ImportError for win32gui on non-Windows platforms
**Solution**: The system now includes cross-platform compatibility. The Windows-specific features will be disabled automatically on Linux/macOS.

**Issue**: Cannot find shape_predictor_68_face_landmarks.dat
**Solution**: Download the file from dlib's website and place it in `gaze_correction_system/lm_feat/`

**Issue**: TensorFlow compatibility errors
**Solution**: Ensure you're using TensorFlow 1.x (not 2.x). Install with: `pip install "tensorflow>=1.14.0,<2.0.0"`

**Issue**: Low frame rate
**Solution**:
- Reduce input resolution
- Use GPU acceleration
- Close other resource-intensive applications

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

When contributing:
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```
[Add your citation information here]
```

## Acknowledgments

- dlib library for facial landmark detection
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## References

- [Add relevant papers or resources here]

---

Last updated: 2025-11-09
