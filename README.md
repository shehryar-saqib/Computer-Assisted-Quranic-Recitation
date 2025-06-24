# 🕌 Quranic Recitation Error Detection (Surah Fatiha)

A machine learning system to detect and correct errors in Quranic recitation, focusing on Surah Al-Fatiha.

## 🚀 Overview

- Detects mispronunciations in Quranic verses  
- Provides feedback on correctness, word order, and mistakes  
- Uses MFCC features and a CNN classifier  
- Speech-to-text powered by Google API

## 🗂️ Dataset

- Sourced from [everyayah.com](https://www.everyayah.com/data) & [dailyayat.com](https://dailyayat.com/)  
- Audio from 66 reciters  
- Preprocessed: MP3 → WAV, 22kHz  
- Augmented via pitch shifting (±3 semitones)

## ⚙️ Model

- CNN with 4 Conv + MaxPooling layers  
- Input: 13 MFCCs  
- Trained for 50 epochs with Adam optimizer  
- Accuracy: **97%**

## 🧪 Error Detection

- Compares transcribed text to reference verse  
- Detects:
  - ✅ Correct recitation  
  - 🔁 Word order issues  
  - ❌ Incorrect words (with index and correction)

## 📈 Results

- High classification accuracy  
- Terminal feedback shows word-level issues (supports Arabic if viewed in Notepad)

## 🛠 Run Instructions

```bash
pip install -r requirements.txt
python train_model.py
python test_recitation.py --file path/to/audio.wav
````

## 🌱 Future Enhancements

* Real-time mic input
* Female recitation support
* Tajweed rules integration
* UI for web/mobile
