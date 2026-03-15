# Hand_Gesture
README.md
markdown# Hand Gesture Recognition using CNN (TensorFlow/Keras)
### Machine Learning Internship — Prodigy InfoTech | Task 4

---

## Overview
This project builds a Convolutional Neural Network (CNN) to recognize
hand gestures from the LEAP Gesture Recognition dataset. The model
classifies 10 different hand gestures from grayscale images using
TensorFlow and Keras.

---

## Internship Details
| Field        | Details                        |
|--------------|--------------------------------|
| Organization | Prodigy InfoTech               |
| Role         | Machine Learning Intern        |
| Task         | Task 4 — Hand Gesture Recognition using CNN |

---

## Dataset
- **Name**   : LEAP Gesture Recognition Dataset
- **Source** : [Kaggle — gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Classes** : 10 hand gestures
  - 01_palm
  - 02_l
  - 03_fist
  - 04_fist_moved
  - 05_thumb
  - 06_index
  - 07_ok
  - 08_palm_moved
  - 09_c
  - 10_down
- **Subjects** : 10 subjects
- **Total Images** : ~20,000 grayscale images

---

## Project Structure
```
hand-gesture-recognition/
│
├── gesture_recognition.ipynb   # Main Jupyter Notebook
├── best_model.keras             # Saved best model
├── gesture_model.keras          # Final saved model
├── training_history.png         # Accuracy & Loss plots
├── confusion_matrix.png         # Confusion matrix plot
├── sample_images.png            # Sample images per class
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## Model Architecture
```
Input (64x64x1)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, softmax)
```

---

## Results
| Metric        | Value     |
|---------------|-----------|
| Test Accuracy | ~99%      |
| Optimizer     | Adam      |
| Loss          | Categorical Crossentropy |
| Epochs        | 30 (EarlyStopping) |

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ganesmpsmg/hand-gesture.git
cd hand-gesture

```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("gti-upm/leapgestrecog")
print("Dataset path:", path)
```

### 4. Run the Notebook
```bash
jupyter notebook hand_gesture.ipynb
```


## Technologies Used
- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- KaggleHub

---

## Author
**Ganesh MP**
Machine Learning Intern — Prodigy InfoTech
- GitHub  : [ganeshmpsmg](https://github.com/ganesmpsmg/ganesmpsmg.github.io)


## License
This project is for educational purposes as part of the
Prodigy InfoTech ML Internship Program.



