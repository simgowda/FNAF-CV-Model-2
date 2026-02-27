to run demo:

only works in python 3.10, and have to use old version 0.10.5 of mediapipe
in command line:

Clone repository
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mediapipe==0.10.5
pip install opencv-python pillow numpy
python webcam_gesture_demo.py
