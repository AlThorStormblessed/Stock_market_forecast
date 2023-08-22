import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def num_of_days(day, month, year):
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0
    days += (year - 2012) * 365
    days += sum(months[:month - 1])
    days += day

    return days

@app.route('/')
def home():
    return render_template('Stock.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(num_of_days(int_features[0], int_features[1], int_features[2]))]
    prediction = model.predict([final_features])

    output = round(prediction[0], 2)

    return render_template('Stock.html', prediction_text=f'Stock price should be {output}')

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([num_of_days(list(data.values())[0], list(data.values())[1], list(data.values())[2])])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)