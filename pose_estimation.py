"""
Author: Melisa Sever
Date: May 12, 2025
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from math import acos, degrees

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (landmarks)
    Points should be in the format [x, y]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = degrees(acos(cosine_angle))
    
    return angle

# Function to classify pose based on landmarks
def classify_pose(landmarks, image_height):
    """
    Improved pose classification for sitting, arm raising, and knee lifting
    Returns the name of the detected pose
    """
    if landmarks is None:
        return "No pose detected"
    
    try:
        # Get relevant landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Get visibility scores for important landmarks
        left_wrist_vis = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
        right_wrist_vis = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
        
        # Calculate angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # ---- IMPROVED DETECTION LOGIC ----
        
        # 2. Now check arm raising (only if knees aren't raised)
        # Improved arm raise detection - look for clear arm positions 
        # above shoulder with good visibility
        arm_vis_threshold = 0.7
        arm_height_threshold = 0.08  # How much higher the wrist needs to be than shoulder
        
        left_arm_raised = (left_wrist[1] < left_shoulder[1] - arm_height_threshold) and \
                          (left_wrist_vis > arm_vis_threshold)
                          
        right_arm_raised = (right_wrist[1] < right_shoulder[1] - arm_height_threshold) and \
                           (right_wrist_vis > arm_vis_threshold)
        
        if left_arm_raised and right_arm_raised:
            return "Both Arms Raised"
        elif left_arm_raised:
            return "Left Arm Raised"
        elif right_arm_raised:
            return "Right Arm Raised"
        
        # 3. We're not using "Arms Behind Back" detection as it causes false positives
        
        # 4. Improved sitting/standing detection
        # Calculate relative positions
        ankle_y_avg = (left_ankle[1] + right_ankle[1]) / 2
        hip_y_avg = (left_hip[1] + right_hip[1]) / 2
        shoulder_y_avg = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Calculate the vertical distance between hips and ankles
        hip_ankle_dist = hip_y_avg - ankle_y_avg
        
        # Check vertical alignment (typical standing posture)
        vertical_alignment = (abs(left_hip[0] - left_shoulder[0]) < 0.15) or \
                            (abs(right_hip[0] - right_shoulder[0]) < 0.15)
        
        # Better standing detection - vertical alignment between shoulders, hips, and ankles
        # and appropriate spacing between body parts
        if vertical_alignment and hip_ankle_dist < 0:
            return "Standing"
            
        # Sitting detection - hips are lower in the frame
        if (hip_y_avg > 0.65) and (hip_ankle_dist > -0.2):
            return "Sitting"
        
        # If none of our specific poses match
        return "Other Pose"
        
    except Exception as e:
        return f"Pose classification error: {str(e)}"

def main():
    # Access webcam
    cap = cv2.VideoCapture(0)
    
    # Variables for FPS calculation
    prev_time = 0
    curr_time = 0
    
    # Create a directory to save images
    save_dir = "pose_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Counter for image naming
    image_counter = 1
    save_message = ""
    save_message_timeout = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break
        
        # Flip the image horizontally for a more intuitive mirror view
        image = cv2.flip(image, 1)
        
        # Get image dimensions
        image_height, image_width, _ = image.shape
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect poses
        results = pose.process(image_rgb)
        
        # Draw pose landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Classify the pose
            pose_class = classify_pose(results.pose_landmarks.landmark, image_height)
            
            # Display pose classification with high visibility
            cv2.putText(image, f"Pose: {pose_class}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Extract and display angles for key joints
            landmarks = results.pose_landmarks.landmark
            
            # Left elbow angle
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Display left elbow angle
            cv2.putText(image, f"L. Elbow: {int(left_elbow_angle)} deg", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Right elbow angle
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Display right elbow angle
            cv2.putText(image, f"R. Elbow: {int(right_elbow_angle)} deg", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display additional elbow information more prominently on the left side
            if left_elbow_angle < 90:
                cv2.putText(image, "Left Elbow Bent", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(image, "Left Elbow Straight", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            if right_elbow_angle < 90:
                cv2.putText(image, "Right Elbow Bent", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(image, "Right Elbow Straight", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else:
            cv2.putText(image, "No pose detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(image, f"FPS: {int(fps)}", (10, image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display save confirmation message
        if save_message and time.time() < save_message_timeout:
            cv2.putText(image, save_message, (10, image.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            save_message = ""
        
        # Display the resulting frame
        cv2.imshow('Pose Estimation', image)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('s'):  # Save image on 's'
            if image is not None:
                filename = os.path.join(save_dir, f"pose_{image_counter}.png")
                cv2.imwrite(filename, image)
                save_message = f"Saved: {filename}"
                save_message_timeout = time.time() + 2
                image_counter += 1
                print(save_message)
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()


