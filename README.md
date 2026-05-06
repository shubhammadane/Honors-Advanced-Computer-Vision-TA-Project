# GestureSpeak 🤟
### Real-Time Sign Language Recognition System

A real-time computer vision system that recognizes hand gestures and sign language letters, converting them into spoken text — built entirely with free, open-source tools.

---

## Demo
- Show ASL hand signs to webcam → letters form words → system speaks aloud
- Gesture mode not available in this version (HaGRID dataset not trained)

---

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| OpenCV | Webcam capture & display |
| MediaPipe | Hand landmark detection |
| TensorFlow/Keras | Gesture classification model |
| pyttsx3 | Offline text-to-speech |
| NumPy / Pandas | Data processing |
| Scikit-learn | Train/test splitting |

---

## Project Structure
```
GestureSpeak/
├── dataset/              ← Download datasets here (see below)
├── model/                ← Saved models (auto-created after training)
├── data/                 ← Processed numpy arrays (auto-created)
├── preprocess_data.py    ← Extract landmarks from dataset images
├── train_model.py        ← Train gesture & alphabet models
├── real_time_detection.py← Main app: webcam + prediction + TTS
├── text_to_speech.py     ← Offline TTS module
├── utils.py              ← Shared helpers, labels, paths
├── requirements.txt      ← All dependencies
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/gayatri-d11/GestureSpeak.git
cd GestureSpeak
```

### 2. Create virtual environment with Python 3.10
```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Download datasets

**Alphabet Mode (ASL):**
- https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Extract into `dataset/alphabet/`

> Note: HaGRID gesture dataset not used in this version.

### 5. Download MediaPipe hand model
```bash
python -c "from utils import download_hand_model; download_hand_model()"
```

### 6. Preprocess data
```bash
python preprocess_data.py
```

### 7. Train models
```bash
python train_model.py
```

### 8. Run the app
```bash
python real_time_detection.py
```

---

## Controls
| Key | Action |
|-----|--------|
| M | Toggle between Alphabet / Gesture mode |
| S | Speak current sentence immediately |
| C | Clear all text |
| Q | Quit |

---

## Model Performance
| Model | Accuracy |
|-------|---------|
| Alphabet (ASL A-Z) | 99.44% |
| Gesture (HaGRID 18 classes) | Not trained (HaGRID dataset not used) |

---

## How It Works
```
Webcam → MediaPipe Hand Landmarks (63 features)
       → Neural Network (256→128→64→output)
       → Predicted Label
       → Text Buffer
       → pyttsx3 speaks aloud
```

---

## Future Enhancements
- Two-hand gesture support
- Word prediction / autocomplete
- GUI with Tkinter
- Support for ISL (Indian Sign Language)
- Mobile deployment with TensorFlow Lite

---
## Demo Video

[![Click to watch demo](https://drive.google.com/thumbnail?id=1HOs4PpiBdQbzQCTggblnAiG8Elr7mEF7)](https://drive.google.com/file/d/1HOs4PpiBdQbzQCTggblnAiG8Elr7mEF7/view?usp=drivesdk)

<img width="648" height="511" alt="Screenshot 2026-05-07 005437" src="https://github.com/user-attachments/assets/88e2c661-80b2-481f-9a08-db5f832ff9a9" />



## Developed by : Gayatri Dabare  &  Aditya Raut

