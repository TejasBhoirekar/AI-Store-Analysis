# AI Store Analysis

AI Store Analysis is a machine learning project that predicts sales for a superstore using a **Random Forest Regressor**. It also includes a simple **Flask-based web UI** to interact with the model and get real-time predictions.  

---

## Features

- Predicts sales for products based on historical data.
- Implements a **Random Forest Regressor** for accurate predictions.
- Provides a **Flask web interface** for easy user interaction.
- Handles CSV datasets for training and testing.

---

## Installation

Follow these steps to set up the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/TejasBhoirekar/AI-Store-Analysis.git
cd AI-Store-Analysis
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
pip install -r requirements.txt
python app.py  #run the flask app
