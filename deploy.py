import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import joblib
import string

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def load_data():
    vector = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')
    return vector, model

def prediction(input_text):
    vector, model = load_data()
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__=="__main__":
    st.title('Fake News Detector')
    input_text = st.text_input('Enter news Article')

    if input_text:
        text = clean_text(input_text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text= text.lower()
        text = [stemmer.stem(word) for word in text.split(' ') if not word in stop_words]
        text = ' '.join(text)
        pred = prediction(text)
        if pred == 0:
            st.write('The News is Fake')
        else:
            st.write('The News Is Real')