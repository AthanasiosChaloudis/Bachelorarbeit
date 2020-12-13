import numpy as np
import pandas as pd
from flask import Flask, render_template, request, make_response
import pickle
from Recommender import recommender

app = Flask(__name__, template_folder='templates')

model = pickle.load(open("save.pkl", "rb"))
cols = ["title", "Correlation", "rating_counts"]

@app.route("/")
def index():
    return render_template('formsWp.html')

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return

@app.route("/test", methods=["POST"])
def test():
    return recommender("Dark Knight Rises, The (2012)")

if __name__ == '__main__':
     app.run(debug="true", host="0.0.0.0", port=5000)