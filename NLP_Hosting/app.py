# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from spellchecker import SpellChecker
import re
import string
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

tweet = "Kentucky tornadoes' death toll reaches 50"
model = None

@app.route('/')
def my_form():
    return render_template('my-form.html')



@app.route('/', methods=['POST'])
def my_form_post():

    #getting the tweet
    tweet = request.form["tweet"]
    prediction = get_prediction(tweet)
    return "The predicted probability of there being a disaster is: " + prediction[0][0]


def get_prediction(tweet):
    cleaned_tweet = clean_tweet(tweet)
    tweet_list = tweet_as_list(cleaned_tweet)

    the_corpus = get_corpus()
    MAX_LEN=50
    tokenizer_obj=Tokenizer()
    tokenizer_obj.fit_on_texts(the_corpus)
    
    the_corpus.append(tweet_list[0])
    sequences = tokenizer_obj.texts_to_sequences(the_corpus)

    tweet_pad = pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post') 
    predict_me = to_predict(tweet_pad)

    #we only need to load in the model the first time
    if model ==  None:
        get_model()
    
    prediction = model.predict(predict_me)
    return prediction[0][0]        

def get_model():
    model = pickle.load(open("nlp_model.sav", 'rb'))
    return model

def to_predict(tweet_pad):
  prediction = tweet_pad[-1]
  reshaped = np.reshape(prediction, (1,50))
  return reshaped

def get_corpus():
    corpus = pickle.load(open("the_corpus.sav", 'rb'))

###formatting methods###
def tweet_as_list(tweet):
  tweet_list = []
  split_words = tweet.split(" ")
  words=[word.lower() for word in split_words if((word.isalpha()==1))]
  tweet_list.append(words)
  return tweet_list


###cleaning methods###
def clean_tweet(tweet):
  clean_tweet = remove_URL(tweet)
  clean_tweet = remove_html(clean_tweet)
  clean_tweet = remove_emoji(clean_tweet)
  clean_tweet = correct_spellings(clean_tweet)
  clean_tweet = remove_punct(clean_tweet)
  return clean_tweet


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

