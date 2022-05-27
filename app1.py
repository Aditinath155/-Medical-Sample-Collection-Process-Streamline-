# -*- coding: utf-8 -*-
"""

"""

# flask app
# importing libraries
import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

from markupsafe import escape
# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('model_1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    final_features = [x for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    
    if prediction == 'Y':
       output='Sample Delivered on Time'
    else:
       output='Sample Not Delivered on Time'
       
    return render_template('index.html', output=output )

if __name__ == '_main_':
    app.run(debug=True)