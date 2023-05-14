from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

model = load(r"C:\Users\xps\Desktop\M1(IAAD)\S2\NLP_DataMining\Fake_news_detection\projets\prediction\svm_model.joblib")

text = "helloz"
vectoriser=TfidfVectorizer()
transform_text=vectoriser.fit_transform([text])

probabilities = model.predict_proba(transform_text)[0]


