from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)

# Load the saved model
model = joblib.load('boston_lr_model.pkl')

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect the input values from the form.
            # Assuming you need exactly the same number of features as used in training.
            # For demonstration, assume the form contains 'feature1', 'feature2', ... 'featureN'
            # Use the feature names from the Boston dataset.
            feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
                             'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
                             'PTRATIO', 'B', 'LSTAT']
            # Collect features in order:
            input_features = [float(request.form.get(feat)) for feat in feature_names]
            # Convert to numpy array and reshape for prediction
            input_array = np.array(input_features).reshape(1, -1)
            # Get prediction from the model
            prediction = model.predict(input_array)[0]
            prediction = round(prediction, 2)
            return render_template('index.html', prediction=prediction, feature_names=feature_names)
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        # Render the form on a GET request
        return render_template('index.html')

# Define a route for API-based prediction (optional)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        # Expecting JSON with the same feature order 
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
                         'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
                         'PTRATIO', 'B', 'LSTAT']
        input_features = [float(data.get(feat)) for feat in feature_names]
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    # Run the app in debug mode
    app.run(debug=True)