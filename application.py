from flask import Flask, request, jsonify
import pickle

# Initialize the Flask application
application = app = Flask(__name__)

# Load the pre-trained model and vectorizer
def load_model():
    global loaded_model, vectorizer

    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

@application.route('/')
def home():
    return "Fake News Detection API is running."

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Please use POST to send data"

    load_model()

    try:
        # get text from the request
        input_text = request.get_json()
        
        if not input_text:
            return jsonify({'error': 'No input'}), 400
        
        print(f"Input: {input_text}")
        
        # predict using provided model
        predictions = []
        for i in input_text:
            vectorized_input = vectorizer.transform([i])
            prediction = loaded_model.predict(vectorized_input)[0]
        
            if prediction == 'FAKE':
                predictions.append(1)
            elif prediction == 'REAL':
                predictions.append(0)
            else:
                raise ValueError(f"Unexpected prediction value: {prediction}")
        
        print("Prediction:", predictions)
        
        return jsonify({'input': input_text, 'prediction': predictions}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "An error occurred during prediction"}), 500

if __name__ == '__main__':
    application.run(port=5000, debug=True)