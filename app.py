from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model pipeline
model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        # Extract the relevant features from the request
        features = [
            data['Car'],
            data['Location'],
            data['Fuel'],
            data['Transmission'],
            data['ODO'],
            data['Age'],  # Use Age instead of Year
            data['Mileage'],
            data['Engine'],
            data['Power'],
            data['Seats'],
            data['CP']
        ]

        # Convert features to DataFrame
        df = pd.DataFrame([features],
                          columns=['Car', 'Location', 'Fuel', 'Transmission', 'ODO', 'Age', 'Mileage', 'Engine',
                                   'Power', 'Seats', 'CP'])
        print("DataFrame:\n", df)

        # Log feature values to debug
        for col in df.columns:
            print(f"{col}: {df[col].values}")

        # Make prediction
        prediction = model.predict(df)
        print("Prediction:", prediction)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
