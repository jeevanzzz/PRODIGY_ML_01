from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Load the test data
with open('test_data.pkl', 'rb') as f:
    X_test, y_test = joblib.load(f)

@app.route('/')
def home():
    return render_template('Homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gr_liv_area = float(request.form['GrLivArea'])
        bedroom_abv_gr = int(request.form['BedroomAbvGr'])
        full_bath = int(request.form['FullBath'])

        input_features = pd.DataFrame([[gr_liv_area, bedroom_abv_gr, full_bath]],
                                      columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])

        predicted_price = model.predict(input_features)[0]

        return render_template('Result.html',
                               prediction=round(predicted_price, 2),
                               sqft=gr_liv_area,
                               bedrooms=bedroom_abv_gr,
                               bathrooms=full_bath)

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
