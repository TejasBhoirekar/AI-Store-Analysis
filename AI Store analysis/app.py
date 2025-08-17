from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Define the column names exactly as used during training
COLUMN_NAMES = [
    'item_weight',
    'item_fat_content',
    'item_visibility',
    'item_type',
    'item_mrp',
    'outlet_establishment_year',
    'outlet_size',
    'outlet_location_type',
    'outlet_type'
]

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    try:
        # Collect form data and convert to float
        data = [float(request.form[col]) for col in COLUMN_NAMES]

        # Convert to DataFrame with correct column names
        X = pd.DataFrame([data], columns=COLUMN_NAMES)

        # Load the scaler and transform
        scaler_path = r'/Users/tejasbhoirekar/Desktop/AI-Superstore-Analysis-main/models/sc.sav'
        sc = joblib.load(scaler_path)
        X_std = sc.transform(X)

        # Load the trained model and make prediction
        model_path = r'/Users/tejasbhoirekar/Desktop/AI-Superstore-Analysis-main/models/rf.sav'
        model = joblib.load(model_path)
        Y_pred = model.predict(X_std)

        # Render result template
        return render_template('result.html', prediction=float(Y_pred[0]))

    except KeyError as e:
        return f"Form field missing: {e}", 400
    except ValueError as e:
        return f"Invalid input: {e}", 400
    except Exception as e:
        return f"Something went wrong: {e}", 500


if __name__ == "__main__":
    app.run(debug=True, port=9457)