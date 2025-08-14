from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Datasets/Cleaned_data.csv')
pipe = pickle.load(open('models/RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bathroom'))
    sqft = request.form.get('sqft')

    input = pd.DataFrame([[locations,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])

    prediction = pipe.predict(input)[0]*1e5

    return str(np.round(prediction,2))


