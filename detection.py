import cv2   # for handling video streams.
import mediapipe as mp   #Used for facial landmark detection.
import numpy as np    #Used for mathematical operations like computing distances.
import time     #Used for tracking time for drowsiness conditions.
from pygame import mixer   #Used for playing an alarm sound when drowsiness is detected.

# The mixer module is used to handle audio playback .
# This must be called before loading or playing any sound.
mixer.init()
mixer.music.load(r"E:\\programming\\Drowsiness detection\\music.wav")

# Initialize MediaPipe Face Mesh

# mp.solutions.face_mesh provides pre-trained models for detecting 468 facial landmarks from an image or a video frame.
# Here, we create a reference mp_face_mesh to access face mesh functionalities.
mp_face_mesh = mp.solutions.face_mesh

'''
initializes the Face Mesh model with two important parameters:
confidence refers to the model's certainty in its predictions
min_detection_confidence=0.5: The minimum confidence (50%) required to detect a face in the frame.
min_tracking_confidence=0.5: The minimum confidence (50%) required for the model to continue tracking
'''
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # mp.solutions.drawing_utils provides tools to draw landmarks on an image.


# Drowsiness Detection Thresholds
EYE_AR_THRESH = 0.25  #If the EAR falls below 0.25, the eyes are considered closed.
EYE_AR_CONSEC_FRAMES = 20  #If the eyes remain closed for 20 consecutive frames, an alarm is triggered.
MOUTH_AR_THRESH = 0.75  #If the MAR goes above 0.75, the person is yawning.
MOUTH_AR_CONSEC_FRAMES = 15  #If yawning continues for 15 consecutive frames, it is considered drowsiness.

# Define Eye & Mouth Landmarks. These landmark indices correspond to specific facial key points provided by the MediaPipe Face Mesh model.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312, 87, 317]


# Function to calculate aspect ratio

'''
landmarks: An array of facial landmark points detected by MediaPipe.
indices: The specific landmark indices of the eye or mouth to calculate the ratio.
'''
def aspect_ratio(landmarks, indices):
    try:
        # landmarks[indices[1]] gives one facial landmark (a point with x, y coordinates).
        # np.linalg.norm() calculates the Euclidean distance (magnitude) between two points in an n-dimensional space.
        A = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])  #Distance between top and bottom points
        B = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])  #Distance between middle-top and middle-bottom points
        C = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])  #Distance between left and right corners
        return (A + B) / (2.0 * C) #Calculating the Aspect Ratio using formula
    #If the provided indices are incorrect or out of bounds, except block will returns 0 instead of crashing.
    except IndexError:
        return 0  

# Initialize webcam
cap = cv2.VideoCapture(0) #to capture live video through the primary camera of the system 
time.sleep(1.0)   #pauses the execution for 1 second to allow the webcam to warm up before capturing frames.

# Counters for drowsiness detection
eye_counter = 0    #tracks the number of consecutive frames where the eyes remain closed.
yawn_counter = 0   #Counts consecutive frames where the mouth is open.
yawn_sequence = 0  #Counts the total number of yawns within a short time 
alarm_on = False   #A flag to track whether the alarm is currently on/off
last_yawn_time = 0  #Stores the time when the last yawn was detected.
alarm_start_time = 0  #Stores the time when the alarm started.

while cap.isOpened():  #Checks if the webcam is available.
    ret, frame = cap.read()  # Reads a frame from the webcam. ret is True if the frame is successfully read.+
    if not ret: #If no frame is captured, exit the loop.
        break

    frame = cv2.flip(frame, 1) #mirrors the image
    h, w, _ = frame.shape  #returns height (h), width (w), and number of color channels (_).
    '''
     _ (underscore) is used for ignoring the third value (channels, typically 3 for RGB)
     height (h) and width (w) are needed to convert normalized facial landmarks (values between 0 and 1) into pixel coordinates.
    '''
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #coverting bgr frame to rgb frame
    '''
    OpenCV loads images in BGR format.
    MediaPipe (which is used for face detection) expects RGB format.
    This conversion ensures the correct color format for face detection.
    '''

    results = face_mesh.process(rgb_frame)
    '''
    process() is a method from MediaPipe's Face Mesh module.
    It takes an image as input and returns facial landmarks if a face is detected.
    '''

    if results.multi_face_landmarks: #if face is detected multi_face_landmarks is an attribute of the results it contains detected facial landmarks if at least one face is found
        for face_landmarks in results.multi_face_landmarks:  # Iterate Over Each Detected Face
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            '''
            Converting Normalized Coordinates to Pixel Coordinates
            lm.x → X-coordinate (horizontal position) in a normalized range [0,1]. lly, lm.y (vertical position)
            Multiply lm.x by image width w → Converts it to real X-position in pixels. lly lm.y with height h
            Storing all the (x, y) pixel coordinates in a NumPy Array
            '''

            # drawing facial landmarks and contours on the frame using MediaPipe's draw_landmarks function.
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            '''
            mp_drawing → MediaPipe's drawing_utils, used to draw landmarks on an image.
            draw_landmarks() → Draws facial landmarks and their connections (contours).
            mp_face_mesh.FACEMESH_CONTOURS - Specifies which facial connections (contours) to draw.
            mp_drawing.DrawingSpec() - Customizes how landmarks are drawn.
            '''

            #checking if total number of detected landmarks on the face is less than equal to highest index from predefined landmark lists (eyes & mouth) as not all landmarks are detected, leading to index errors when accessing landmark points.
            if len(landmarks) >= max(max(LEFT_EYE), max(RIGHT_EYE), max(MOUTH)):
                eye_AR = (aspect_ratio(landmarks, LEFT_EYE) + aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
                '''
                aspect_ratio(landmarks, LEFT_EYE) → Computes the aspect ratio for the left eye.
                aspect_ratio(landmarks, RIGHT_EYE) → Computes the aspect ratio for the right eye.
                '''
                mouth_AR = aspect_ratio(landmarks, MOUTH)
                '''
                Uses the aspect_ratio() function to calculate how open the mouth is.
                '''
            #If the detected face does not have enough landmarks, the loop skips processing and moves to the next frame.
            else:
                continue


            # Debugging: Print EAR & MAR values
            # Display EAR & MAR on the screen
            cv2.putText(frame, f"EAR: {eye_AR:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #print EAR on frame 
            cv2.putText(frame, f"MAR: {mouth_AR:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"EAR: {eye_AR:.2f}, MAR: {mouth_AR:.2f}") #Prints EAR and MAR values in the terminal/log for debugging.


            # detects if the eyes are closed for a certain duration and triggers an alarm when necessary.
            if eye_AR < EYE_AR_THRESH:  #If EAR is less than the threshold, it means the eyes are closing or closed
                eye_counter += 1  #Each frame where eyes are closed, eye_counter increases by 1
                if eye_counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "EYES CLOSED!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) #Displays "EYES CLOSED!" on the webcam feed.
                    if not alarm_on:   #checking if alarm is on or not if off then open it
                        print("ALARM: Eyes Closed!")
                        mixer.music.play(-1)   #plays the alarm -1 makes it loop
                        alarm_on = True   #setting alarm on varible true so it doesn’t restart unnecessarily
                        alarm_start_time = time.time()   #time.time() Returns the current time in seconds to track when the alarm started
            else:
                eye_counter = 0 #resetting eye counter

            # Yawning detection with 4-yawn condition
            current_time = time.time()  # taking systems current time for tracking the time difference between yawns
            if mouth_AR > MOUTH_AR_THRESH: #Detect if the Mouth is Open 
                yawn_counter += 1 #increasing yawn counter by 1
                if yawn_counter >= MOUTH_AR_CONSEC_FRAMES: #If the mouth stays open for MOUTH_AR_CONSEC_FRAMES (e.g., 15 frames), it confirms a full yawn.
                    cv2.putText(frame, "YAWNING DETECTED!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) #Shows a "Yawning Detected!" warning on the screen.

                    '''
                    If the last yawn was less than 5 seconds ago, increase yawn_sequence count.
                    Otherwise, reset yawn_sequence to 1, starting a new count. 
                    '''
                    if current_time - last_yawn_time < 5:  
                        yawn_sequence += 1
                    else:
                        yawn_sequence = 1  

                    last_yawn_time = current_time   #Updates the timestamp of the last detected yawn so that is new yawn encountered wich check time gap between these 2 
                    print(f"Yawn Count: {yawn_sequence}")  #printing on console

                    if yawn_sequence >= 4 and not alarm_on: #If 4 yawns occur within 5 seconds, an alarm is triggered.
                        print("ALARM: 4 Continuous Yawns Detected!")  #printing on terminal
                        mixer.music.play(-1)   #playing sount
                        alarm_on = True         #setting alarm on variable true
                        alarm_start_time = time.time()   #noting start time of alarm 

                    yawn_counter = 0   #else setting yawn counter to 0
            else:  #Detect if the Mouth is not Open 
                yawn_counter = 0  #setting yawn counter to 0. This ensures that the program only counts consecutive yawns.

                #Reset yawn_sequence After 4 Seconds of No Yawning
                if current_time - last_yawn_time >= 4 and yawn_counter > 0:
                    print("Resetting Yawn Sequence")
                    yawn_sequence = 0

            # Stop the Alarm When Fully Awake that is when eyes are open and yawn counter is 0
            if alarm_on and (eye_counter == 0 and yawn_counter == 0):
                #If the alarm has been playing for more than 1 second, turn it OFF.
                if time.time() - alarm_start_time > 1:
                    print("ALARM OFF")
                    mixer.music.stop()
                    alarm_on = False #Resets alarm_on to False, so it can trigger again if needed.

    #Opens a window named "Drowsiness Detection" and Continuously updates the window with the processed frame containing
    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF #Waits for a key press for 1 millisecond. & 0xFF ensures compatibility across different systems.
    if key == ord("q"): #If the 'q' key is pressed, the program breaks out of the loop
        break

cap.release()   #Releases the webcam (cap), stopping it from capturing video.
cv2.destroyAllWindows() #Closes all OpenCV windows to free up system resources
mixer.music.stop()  #Stops any alarm sound if it is still playing.

print("Program terminated.") #Prints "Program terminated." in the console.