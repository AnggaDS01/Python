from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            sepal_length = request.form.get('sepal_length'),
            sepal_width = request.form.get('sepal_width'),
            petal_length = request.form.get('petal_length'),
            petal_width = request.form.get('petal_width'),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)