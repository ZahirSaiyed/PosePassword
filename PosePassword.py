import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def poseDetect():
    FRAME_WINDOW = st.image([])
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Video Feed
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract Landmarks
            try: 
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                leftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                rightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Calculate angles
                leftAngle = calculate_angle(leftHip, leftShoulder, leftElbow)
                rightAngle = calculate_angle(rightHip, rightShoulder, rightElbow)
                
                # Visualize angles

                if(abs(leftAngle-90) <= 5):
                    cv2.putText(image, "Success!", 
                            tuple(np.multiply(leftShoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127,255,0), 2, cv2.LINE_AA
                            )
                else:
                    cv2.putText(image, str(leftAngle), 
                            tuple(np.multiply(leftShoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                if(abs(rightAngle-90) <= 5):
                        cv2.putText(image, "Success!", 
                            tuple(np.multiply(rightShoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127,255,0), 2, cv2.LINE_AA
                            )
                else:
                    cv2.putText(image, str(rightAngle), 
                                tuple(np.multiply(rightShoulder, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

            except:
                pass
            
            # Render detection 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            #cv2.imshow('Mediapipe Feed', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main_loop():
    st.title("OpenCV Pose Password App")
    st.subheader("This app requires specific poses as passwords")
    st.text("We used OpenCV and Streamlit for this demo")

    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    # FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(0)

    while run:
        # _, frame = camera.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FRAME_WINDOW.image(frame)
        poseDetect()
    else:
        st.write('Stopped')


if __name__ == '__main__':
    main_loop()