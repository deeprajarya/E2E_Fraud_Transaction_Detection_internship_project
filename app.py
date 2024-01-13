from src.Fraud_TX.pipelines.prediction_pipeline import PredictionPipeline, CustomData
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')

def home_page():
    return render_template("form.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("index.html")
    else:
        data=CustomData(
            Time=float(request.form.get('Time')),
            V1=float(request.form.get('V1')),
            V2=float(request.form.get('V2')),
            V3=float(request.form.get('V3')),
            V4=float(request.form.get('V4')),
            V5=float(request.form.get('V5')),
            V6=float(request.form.get('V6')),
            V7=float(request.form.get('V7')),
            V8=float(request.form.get('V8')),
            V9=float(request.form.get('V9')),
            V10=float(request.form.get('V10')),
            V11=float(request.form.get('V11')),
            V12=float(request.form.get('V12')),
            V13=float(request.form.get('V13')),
            V14=float(request.form.get('V14')),
            V15=float(request.form.get('V15')),
            V16=float(request.form.get('V16')),
            V17=float(request.form.get('V17')),
            V18=float(request.form.get('V18')),
            V19=float(request.form.get('V19')),
            V20=float(request.form.get('V20')),
            V21=float(request.form.get('V21')),
            V22=float(request.form.get('V22')),
            V23=float(request.form.get('V23')),
            V24=float(request.form.get('V24')),
            V25=float(request.form.get('V25')),
            V26=float(request.form.get('V26')),
            V27=float(request.form.get('V27')),
            V28=float(request.form.get('V28'))

        )

        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictionPipeline()
        
        preddictions=predict_pipeline.predict(final_data)
        
        result=int(preddictions)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(debug=True, port=5000)
