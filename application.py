import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('ridge.pkl','rb'))
standard_scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=["POST"])
def predict_datapoint():

    Temperature = float(request.form.get('Temperature'))
    RH = float(request.form.get('RH'))
    Ws = float(request.form.get('Ws'))
    Rain = float(request.form.get('Rain'))
    FFMC = float(request.form.get('FFMC'))
    DMC = float(request.form.get('DMC'))
    ISI = float(request.form.get('ISI'))
    Classes = float(request.form.get('Classes'))
    Region = float(request.form.get('Region'))

    new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
    new_data_scaled = standard_scaler.transform(new_data)

    result = ridge_model.predict(new_data_scaled)

    return render_template('home.html', results=result[0])

if __name__ == '__main__':
    app.run(debug=True)