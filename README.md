# Face Emotion Recognition

*Basically Vision, but trained on faces instead of Infinity Stones.*

## Project Overview

This project detects human faces using **OpenCV** and recognizes their **emotions in real time** using a trained deep learning model.

It supports:

* Training with strong data augmentation
* Robust evaluation (accuracy, confusion matrix, classification report)
* Image-based emotion prediction
* Real-time webcam emotion tracking with bounding boxes

## Project Structure

```
face_emotion/
│
├── train_emotion.py        # Train & evaluate emotion model, save metrics and plots
├── predict_emotion.py      # Predict emotions on private images
├── realtime_emotion.py     # Real-time face detection + emotion recognition
│
├── dataset/                # Emotion-class folders (angry, happy, sad, etc.)
├── dataset_private/
│   ├── input_images/
│   └── output_images/
│
├── artifacts/              # Saved models, metrics, plots
```

## Training Highlights

* Stratified **train/validation/test** split
* Strong augmentation (flip, affine, blur, erasing, perspective)
* **Class-weighted loss** to handle imbalance
* Two-phase training:

  1. Frozen backbone (train classifier head)
  2. Fine-tuning with low learning rate
* Early stopping + LR scheduler

## Outputs

* Best & final trained models
* Loss and accuracy plots
* Confusion matrix & classification report
* Emotion-labeled images and real-time webcam overlay

## Dataset

Uses the **[FER dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data)** from Kaggle (7 emotion classes).

## Sample Results
<img width="1534" height="418" alt="Zrzut ekranu 2026-01-30 164724" src="https://github.com/user-attachments/assets/49d1218f-2686-4459-9e1b-db5815216292" />

