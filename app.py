import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from ai import model, feature_extraction


app = Flask(__name__)

ML_model = model.RF_model

@app.route('/')
def home():
    return render_template('./home.html', valor=None)

@app.route("/", methods=['POST'])
def predict():
    url = request.form['url']
    url_data = feature_extraction.feature_extraction(url)

   


    valor = ML_model.predict(url_data)
    valor = 'Legitimo' if valor == 0 else 'Phishing'

    probabilities = ML_model.predict_proba(url_data)[0]
    probabilities = f"{np.max(probabilities) * 100:.1f}"


    return render_template('home.html',
                           url=url, 
                           valor=valor,
                           probabilities=probabilities
                           )


if __name__ == '__main__':
    app.run(debug=True)