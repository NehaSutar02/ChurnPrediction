# ChurnGuard — Multi-Industry Churn Prediction

An interactive machine learning application that predicts customer churn using industry-specific customer data and trained classification models.

## Overview

ChurnGuard is a Streamlit-based machine learning project designed to explore customer churn prediction. It provides a foundation for analyzing customer information, generating churn predictions, and supporting data-driven customer retention decisions.

## Features

- Interactive web interface built with Streamlit
- Python-based machine learning workflow
- Data preprocessing and prediction modules
- Support for sample datasets
- Trained model integration
- Modular and organized project structure

## Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn-compatible workflow, depending on the implemented model
- **Data Processing:** Pandas, NumPy
- **Model Storage:** Joblib
- **Version Control:** Git and GitHub

## Project Structure

```text
ChurnPrediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample/
├── models/
└── src/
    ├── config.py
    ├── make_synthetic.py
    ├── predict.py
    ├── preprocess.py
    └── train.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NehaSutar02/ChurnPrediction.git
cd ChurnPrediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Machine Learning Workflow

1. Load the required customer dataset.
2. Preprocess the input features.
3. Train or load the configured machine learning model.
4. Generate churn predictions.
5. Display prediction results through the Streamlit interface.

## Use Cases

- Customer retention analysis
- Churn risk assessment
- Exploratory customer analytics
- Machine learning experimentation

## Future Improvements

- Add model evaluation metrics and visualizations
- Compare multiple classification algorithms
- Provide industry-specific datasets and model configurations
- Deploy the application online
- Add explainability features for prediction results

## Author

**Neha Sutar**

GitHub: [@NehaSutar02](https://github.com/NehaSutar02)

---

*This project is intended for educational and portfolio purposes. Prediction quality depends on the dataset, model, and validation methodology used.*
