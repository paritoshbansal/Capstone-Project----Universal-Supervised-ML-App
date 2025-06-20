# 🤖 Universal Supervised ML App
An intelligent Streamlit-based AutoML app that automatically detects whether your task is **regression** or **classification**, trains multiple models, evaluates them, and visualizes results — all in one click!

👉 **[Try the Live App](https://capstone-project----universal-supervised-ml-app-6njqtheb6k7s7s.streamlit.app/)**


## 🚀 Features
✅ Auto-detects task type (classification or regression)  
✅ Trains multiple models and selects the best  
✅ Handles outliers (IQR method), missing values, scaling  
✅ Label encoding for classification tasks  
✅ Residuals and performance visualizations  
✅ Upload future datasets to generate predictions  
✅ Export results with one click  

## 📥 Input Format
- ✅ Upload your training data CSV
- ✅ Select the target (dependent) variable
- ✅ (Optional) Upload future data for predictions

## 📤 Output
- ✅ Best model name and score
- ✅ Evaluation plots (e.g., Confusion Matrix, Residual Plot)
- ✅ CSV download of future predictions

## 🛠 Example Use Cases
- ✅ Titanic dataset (classification)
- ✅ House price prediction (regression)
- ✅ Customer churn, loan default, etc.

## 📊 Tech Stack
- [Python 3.8+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- pandas, NumPy, seaborn, matplotlib
