from flask import Flask, render_template, request
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the trained model and vectorizer
loaded_model = pickle.load(open('model_new.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


def fake_news_det(news):
    try:
        # Vectorize the input text using the loaded TF-IDF vectorizer
        vectorized_text = tfidf_vectorizer.transform([news])
        
        # Make prediction
        prediction = loaded_model.predict(vectorized_text)
        
        if prediction[0] == False or prediction[0] == 0:
            return "Prediction: Looking FAKE News"
        else:
            return "Prediction: Looking REAL News"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get('news', '')
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)