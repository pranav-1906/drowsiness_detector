# 💤 Drowsiness Detector

A real-time drowsiness detection system that uses facial landmarks to monitor eye aspect ratio (EAR) and sounds an alarm when drowsiness is detected.

---

## 🔍 Features

- Uses **MediaPipe FaceMesh** to detect facial landmarks.
- Calculates **Eye Aspect Ratio (EAR)** to determine if eyes are closed.
- Sounds an alarm using **Pygame** when drowsiness is detected.
- Runs in real-time using webcam input via **OpenCV**.

---

## 🖥️ Demo

> https://www.linkedin.com/posts/pranav1906_python-computervision-opencv-activity-7354550161072627713-2XHX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFD5jKEBJZyynUi9l5jfTa35m9WBU4PLLJ0

---

## 🛠️ Requirements

Install the required packages using:

```bash
pip install opencv-python mediapipe numpy pygame
```

## 🚀 How to Run

Clone the repository or download the files.

Ensure alarm.mp3 is in the same directory as eyes.py.

Run the main script:

```bash
python eyes.py
```
Allow webcam access. If the system detects your eyes are closed for too long, it will sound an alarm.

## ⚙️ How It Works

The system uses MediaPipe to detect specific facial landmarks around the eyes.

Calculates Eye Aspect Ratio (EAR) — this value drops when eyes are closed.

If EAR remains below a threshold (default: 0.23) for multiple frames, an alarm is triggered to wake the user.

## 📁Project Structure
```bash

├── eyes.py           # Main script
├── alarm.mp3         # Alarm audio file
├── open_ear.csv      # (Optional) EAR data when eyes open
├── closed_ear.csv    # (Optional) EAR data when eyes closed
├── eyes_v2.py        # Alternate version
├── eyes_v3.py        # Alternate version
├── eyes_v4           # Possibly a compiled/binary version
```
