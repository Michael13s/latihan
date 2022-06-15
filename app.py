#!/usr/bin/env python
# coding: utf-8

# In[3]:


from datetime import date
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def wave_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        print(dict(request.form))
        wave_features = dict(request.form).values()
        wave_features = np.array(date(x) for x in wave_features)
        model, scaler = joblib.load("lstm_multivariate_prediksi_berlayar.pkl")
        wave_features = scaler.transform[wave_features]
        print(wave_features)
        result = model.predict(wave_features)
        wave = {
            '0' : 'Dapat Berlayar',
            '1' : 'Tidak Dapat Berlayar' 
        }
        # result = np.round(result, 2)
        result = wave[date(result[0])]
        return render_template('index.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)


# In[ ]:




