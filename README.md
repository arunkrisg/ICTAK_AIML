# Scenary Classification App

## About
This project classifies natural scene images into
6 categories - buildings, forest, glacier,
mountain, sea and street.

## Dataset
Intel Image Classification Dataset from Kaggle

## Models Used
- Task 1 to 5 : Simple CNN model - 71.7% accuracy
- Task 6      : MobileNetV2 Transfer Learning - 88.13% accuracy

## How to Run App
1. Install requirements:
   pip install streamlit tensorflow pillow
2. Run app:
   streamlit run app.py

## Files
- aiml-exit-examination.ipynb    : Complete training notebook
- app.py            : Streamlit application
- final_model.keras : Trained model
- class_names.json  : Class labels