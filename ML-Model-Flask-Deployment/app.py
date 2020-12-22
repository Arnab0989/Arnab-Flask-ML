import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    print("Predict is called")
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    For direct API calls trough request
    Json Sample
    {"experience":2, "test_score":9, "interview_score":6}
    """

    print("Predict API is called")
    data = request.get_json(force=True)
    print(data)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    print("Before app run")
    app.run(debug=True)
    print("main is called")
