import numpy as np 
import pickle
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)
#loading  the file
model_pickle=pickle.load(open("model.pkl","rb"))


@app.route('/')
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]

    features=[np.array(float_features)]
    prediction=model_pickle.predict(features)

    return render_template("index.html",prediction_text=f"The flower species be {prediction}")


if __name__=="__main__":
    app.run(debug=True)

