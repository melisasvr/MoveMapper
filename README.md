# MoveMapper

MoveMapper is a Python-based real-time pose estimation tool that uses MediaPipe to detect and classify human body poses from a webcam feed. It identifies specific poses such as raising one or both arms, sitting, and standing, while providing visual feedback like elbow angles and pose labels. This project is ideal for developers, researchers, or enthusiasts interested in computer vision, human movement analysis, or interactive applications.

## Features
- **Real-Time Pose Detection**: Tracks upper-body poses using MediaPipe Pose.
- **Pose Classification**:
  - Arm raises (left, right, or both).
  - Sitting and standing postures.
  - Other poses (catch-all for unclassified positions).
- **Visual Feedback**:
  - Displays pose labels (e.g., "Left Arm Raised").
  - Shows left and right elbow angles (e.g., "L. Elbow: 170 deg").
  - Indicates elbow state (bent or straight).
  - Renders pose landmarks on the webcam feed.
- **Image Saving**: Saves snapshots of the webcam feed with pose annotations (press 's').
- **Performance**: Displays frames per second (FPS) for real-time monitoring.

## Prerequisites
- **Python**: Version 3.6–3.10 (MediaPipe compatibility).
- **Webcam**: A working webcam connected to your computer.
- **Dependencies**:
  - `mediapipe`
  - `opencv-python`
  - `numpy`

## Installation
1. **Clone the Repository**:
   ```
   git clone https://github.com/<your-username>/MoveMapper.git
   cd MoveMapper
   ```
2. **Install Dependencies**:
  ```
  pip install --upgrade mediapipe opencv-python numpy
```

## Usage
1. Run the Script:
`python move_mapper.py`
- A window will open showing the webcam feed with pose landmarks and annotations.

2. Interact with the Program:
- Pose Detection: Position yourself in the camera frame (shoulders to hips visible) to detect:
- Arm Raises: Raise one or both arms straight up.
- Sitting: Sit down (hips low in frame).
- Standing: Stand upright.
- Save Images: Press s to save a snapshot to the pose_images directory.
- Quit: Press q to exit.

## Output:
- Pose label (top left, e.g., "Pose: Sitting").
- Elbow angles and state (e.g., "L. Elbow: 90 deg", "Left Elbow Bent").
- FPS (bottom left).
- Saved images in pose_images/ (excluded from Git via .gitignore).

## Project Structure
```
MoveMapper/
├── move_mapper.py      # Main pose estimation script
├── pose_images/        # Directory for saved images (ignored by .gitignore)
├── .gitignore          # Excludes pose_images and other files
├── README.md           # Project documentation
```

## Notes
- Privacy: The pose_images directory is ignored by .gitignore to prevent uploading webcam captures to GitHub.
- Webcam Issues: If the webcam fails, try changing cv2.VideoCapture(0) to cv2.VideoCapture(1) or 2 in move_mapper.py.
- Lighting and Positioning: Ensure bright, even lighting and keep your upper body (shoulders to hips) in the camera frame for accurate detection.


## License
- This project is licensed under the MIT License. See the LICENSE file for details.

