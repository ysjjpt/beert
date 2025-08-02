#app.py
from inference import predict

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    text = request.json.get('text', '')
    if text:
        prediction = predict(text)
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=False)

