#!/home/juanxo90/anaconda3/envs/ml_nlp/bin/python python

## to run the app: gunicorn --bind 0.0.0.0:9696 predict:app
import pickle
from flask import Flask
from flask import request
from flask import jsonify

input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:  ## read bites
    dv, model = pickle.load(f_in)  # better way to save it

# %%
app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predcit():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': y_pred,
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

