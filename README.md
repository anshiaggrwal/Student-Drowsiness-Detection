Student Drowsiness Detection

Overview
Student Drowsiness Detection is a real-time monitoring system designed to detect drowsiness in students during study sessions, online classes, or lectures. The system uses computer vision techniques with MediaPipe Face Mesh to track facial landmarks and determine signs of drowsiness based on eye closure and yawning patterns. An alarm is triggered when drowsiness is detected.

Features
Eye Aspect Ratio (EAR) Calculation: Detects prolonged eye closure.
Mouth Aspect Ratio (MAR) Calculation: Identifies frequent yawning.
Real-Time Monitoring: Uses a webcam to continuously track facial expressions.
Alarm System: Plays an alert sound when drowsiness is detected.
Live Landmark Visualization: Displays facial landmark detection in real-time.
Planned Enhancement: Posture Analysis (to track head tilting and slouching).
Installation

Clone the repository:
git clone https://github.com/your-username/Student-Drowsiness-Detection.git
cd Student-Drowsiness-Detection

Create and activate a virtual environment (optional but recommended):
python -m venv drowsiness_env
source drowsiness_env/bin/activate  # For macOS/Linux
drowsiness_env\Scripts\activate     # For Windows

Install dependencies:
pip install -r requirements.txt

Usage
Run the script:
python drowsiness_detection.py

Ensure your webcam is enabled.
The system will analyze your face in real-time.
If signs of drowsiness (closed eyes or yawning) are detected, an alarm will sound.
Press q to exit the program.

Dependencies
OpenCV - For video processing.
MediaPipe - For facial landmark detection.
NumPy - For mathematical operations.
Pygame - For playing alarm sounds.
Install them manually if needed:
pip install opencv-python mediapipe numpy pygame

Future Improvements
Posture Analysis: Detect head tilts and slouching.
Enhanced Alert System: Provide visual warnings in addition to audio.
Data Logging: Track drowsiness patterns over time.

Contribution
If you'd like to contribute, feel free to fork the repository and submit a pull request.

