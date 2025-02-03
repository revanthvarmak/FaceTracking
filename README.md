# Face Tracking using MTCNN and ResNet

This repository implements face tracking and crops video clips of a target face based on a reference image. It uses MTCNN for face detection and InceptionResnetV1 for face recognition. Cropped clips containing the target face are saved as individual video files, and metadata (timestamps and face coordinates) is generated for each clip.

## Key Design Decisions
* **Re-detection Strategy Over DeepSORT:**  
  Instead of using DeepSORT for multi-object tracking, this solution focuses on single object tracking. A re-identification method is used by closely monitoring a region around the previous bounding box with a lower threshold for robust tracking.
  To emphasize the advantages of re-identification, the detection made by the primary detector is colored green and the one made by the secondary detector is colored red.
* **Increasing the reference image dataset:**  
  With only one reference image available, a queue based approach is used to store additional reference embeddings. As faces are recognized over time, their embeddings are added to the queue to make it more robust.

## Requirements
* Python 3.7+
* OpenCV
* PyTorch
* facenet_pytorch
* Pillow
* NumPy

### Install the dependencies with:

```bash
pip install -r requirements.txt
```

### Usage:
1. Clone the repository
```bash
git clone git@github.com:revanthvarmak/FaceTracking.git
```
2.Run the script
```bash
python model.py
```

### Limitations:
1. This implementation cannot detect when the person to be detected turns sideways, since we are using only one reference image. To improve this, we need the images of the person from multiple views
2. Needs more processing time


### Acknowledgments:
Reference video is taken from : https://www.youtube.com/watch?v=WYCo-3pw52o
