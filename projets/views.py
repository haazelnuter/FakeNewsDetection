from django.shortcuts import render
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from django.conf import settings
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def predict(request):
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if text:
            model_path = r"C:\Users\xps\Desktop\M1(IAAD)\S2\NLP_DataMining\Fake_news_detection\projets\prediction\svm_model.joblib"
            vectorizer_path = r"C:\Users\xps\Desktop\M1(IAAD)\S2\NLP_DataMining\Fake_news_detection\projets\prediction\tfidf.pkl"

            try:
                model = load(model_path)
                vectorizer = load(vectorizer_path)

                # Preprocess the text
                preprocessed_text = preprocess_text(text)

                # Apply TF-IDF transformation
                X_new = vectorizer.transform([preprocessed_text])

                probabilities = model.predict_proba(X_new)[0]
                real_prob = probabilities[0]
                fake_prob = probabilities[1]
            except (FileNotFoundError, ValueError, KeyError) as e:
                # Handle any errors that may occur
                error_message = "Error occurred during prediction: {}".format(e)
                return render(request, 'home.html', {'error_message': error_message})

        else:
            fake_prob = None
            real_prob = None
        return render(request, 'home.html', {'fake_prob': fake_prob, 'real_prob': real_prob})

    return render(request, 'home.html')
