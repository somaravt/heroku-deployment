from flask import Flask, jsonify,  request, render_template
import numpy as np
import pandas as pd
from model import get_top_20_recommended_products,get_top_5_products_using_sentiment_analysis

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/recommendations", methods=['POST'])
def recommend():
    if (request.method == 'POST'):
        int_features = [x for x in request.form.values()]
        recommendations =  get_top_20_recommended_products(int_features[0])
        output = get_top_5_products_using_sentiment_analysis(recommendations)
        return render_template('index.html', prediction=output)
    else :
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=2000)