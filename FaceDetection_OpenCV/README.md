# 👤 Face Detection using OpenCV

Detects human faces in images using OpenCV Haar Cascade Classifier.

## Results
- Detects multiple faces in a single image
- Draws green bounding boxes around detected faces
- Works on group photos and individual portraits

## Tech Stack
- Python
- OpenCV

## How to Run
pip install opencv-python
python detect.py

## How it Works
- Loads pre-trained Haar Cascade face detector
- Converts image to grayscale
- Scans image at multiple scales
- Draws green box around each detected face

## Limitations
- Works best on front facing faces
- Struggles with angled faces
- Performance depends on image quality