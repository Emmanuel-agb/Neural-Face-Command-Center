# Neural Face Command Center

A real-time computer vision project that combines **MediaPipe Face Detection**, **MediaPipe Hands**, and **OpenCV** to turn a normal webcam into an interactive **face-command center** with futuristic HUD overlays and gesture-controlled visual modes.

## Overview

This project goes beyond basic face detection by combining:

- real-time face tracking
- hand gesture recognition
- dynamic HUD overlays
- gesture-controlled visual modes
- particle effects for a more cinematic webcam experience

With different hand signs, the system can trigger effects like lock-on mode, identity scan, aura shield, portal visuals, and focus assist directly from the webcam feed.

## Features

- Real-time face detection using MediaPipe
- Real-time hand tracking using MediaPipe Hands
- Gesture recognition for multiple hand signs
- Futuristic HUD box around the face
- Visual effects and particles for mode changes
- Live webcam interaction with OpenCV
- Screenshot capture support

## Gesture Controls

- **PALM** → Aura Shield
- **FIST** → Lock-On Mode
- **PEACE** → Identity Scan
- **PINCH** → Portal Channel
- **POINT** → Focus Assist

## Keyboard Controls

- **q** or **Esc** → Quit the app
- **s** → Save a screenshot
- **r** → Reset particles

## Tech Stack

- Python
- OpenCV
- MediaPipe Face Detection
- MediaPipe Hands
- NumPy

## Project Structure

```bash
Neural-Face-Command-Center/
├── neural_face_command_center.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Emmanuel-agb/Neural-Face-Command-Center.git
cd Neural-Face-Command-Center
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python neural_face_command_center.py
```

## How It Works

The webcam feed is captured with OpenCV. MediaPipe Face Detection identifies the face and key points, while MediaPipe Hands tracks hand landmarks and classifies hand gestures. Based on the detected gesture, the system switches visual modes and overlays an animated command-center style interface on the live video stream.

## Why This Project Stands Out

Instead of building a basic face detector or another standard face mesh demo, this project focuses on creating a **more immersive real-time experience** by combining face detection, gesture interaction, visual effects, and a futuristic interface.

It is designed to be both:

- a strong portfolio project
- a visually engaging LinkedIn/GitHub showcase

## Future Improvements

- Add voice-controlled mode switching
- Add cyber mask overlay around the face
- Add recording mode for demo videos
- Add multi-person tracking
- Add sound effects for each gesture mode
- Add emotion-aware visual responses

## Demo Idea

A great demo flow for this project is:

1. Show real-time face detection
2. Raise a hand and switch between gesture modes
3. Trigger lock-on, identity scan, and portal effects
4. Save a clean screenshot or record a short video for LinkedIn/GitHub

## Repository Description

**Short Description:**

Real-time face detection and hand gesture recognition system with futuristic HUD overlays using Python, OpenCV, and MediaPipe.

## Author

Built by **Emmanuel Mawuli Agbator**
