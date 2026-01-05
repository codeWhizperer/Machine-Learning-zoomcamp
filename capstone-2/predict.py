import pickle
from flask import Flask, request, jsonify

THRESHOLD = 0.402

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('xgb_model.bin', 'rb') as f:
    model = pickle.load(f)

app = Flask('trader_classifier')

@app.route('/predict', methods=['POST'])
def predict():
    trader = request.get_json()
    X = dv.transform([trader])
    proba = model.predict_proba(X)[0, 1]

    return jsonify({
        'predicted_proba': float(proba),
        'predicted_target': int(proba >= THRESHOLD)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)
