from django.shortcuts import render
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

def predict(request):
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if text:
            model = load(r"C:\Users\xps\Desktop\M1(IAAD)\S2\NLP_DataMining\Fake_news_detection\projets\prediction\svm_model.joblib")
            vectorizer = load(r"C:\Users\xps\Desktop\M1(IAAD)\S2\NLP_DataMining\Fake_news_detection\projets\prediction\tfidf.pkl")
            X_new = vectorizer.transform([text])
            probabilities = model.predict_proba(X_new)[0]
            real_prob = probabilities[0]
            fake_prob = probabilities[1]
        else:
            fake_prob = None
            real_prob = None
        return render(request, 'home.html', {'fake_prob': fake_prob, 'real_prob': real_prob})
    return render(request, 'home.html')
