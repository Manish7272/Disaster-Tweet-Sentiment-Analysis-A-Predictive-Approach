# importing Libraries

import streamlit as st
import PIL
from PIL import Image
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import numpy as np
import pandas as pd
import nltk

try:                                                                         # Check if wordnet is installed
    nltk.find("corpora/wordnet.zip")          
except LookupError:
    nltk.download('wordnet')

# ----------------------------------------------------------------------------------
# read files
try:
    acronyms_dict, contractions_dict, stops
except NameError:
    acronyms_dict = pd.read_json("acronym.json", typ = "series")
    contractions_dict = pd.read_json("contractions.json", typ = "series")
    stops = list(pd.read_csv('stopwords.csv').values.flatten())

# ----------------------------------------------------------------------------------
# Defining tokenizer
regexp = RegexpTokenizer("[\w']+")

# preprocess Function
def preprocess(text):
    
    text = text.lower()                                                                                        # lowercase
    text = text.strip()                                                                                        # whitespaces
    
    # Removing html tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)                                                                                 # html tags
    
    # Removing emoji patterns
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'', text)                                                                         # unicode char
    
    # Removing urls
    http = "https?://\S+|www\.\S+" # matching strings beginning with http (but not just "http")
    pattern = r"({})".format(http) # creating pattern
    text = re.sub(pattern, "", text)                                                                            # remove urls
    
    # Removing twitter usernames
    pattern = r'@[\w_]+'
    text = re.sub(pattern, "", text)                                                                            # remove @twitter usernames
    
    # Removing punctuations and numbers
    punct_str = string.punctuation + string.digits
    punct_str = punct_str.replace("'", "")
    punct_str = punct_str.replace("-", "")
    text = text.translate(str.maketrans('', '', punct_str))                                                     # punctuation and numbers
    
    # Replacing "-" in text with empty space
    text = text.replace("-", " ")                                                                               # "-"
    
    # Substituting acronyms
    words = []
    for word in regexp.tokenize(text):
        if word in acronyms_dict.index:
            words = words + acronyms_dict[word].split()
        else:
            words = words + word.split()
    text = ' '.join(words)                                                                                       # acronyms
    
    # Substituting Contractions
    words = []
    for word in regexp.tokenize(text):
        if word in contractions_dict.index:
            words = words + contractions_dict[word].split()
        else:
            words = words + word.split()
    text = " ".join(words)                                                                                       # contractions
    
    punct_str = string.punctuation
    text = text.translate(str.maketrans('', '', punct_str))                                                     # punctuation again to remove "'"
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in regexp.tokenize(text)])                             # lemmatize
    
    # Stopwords Removal
    text = ' '.join([word for word in regexp.tokenize(text) if word not in stops])                              # stopwords
    
    # Removing all characters except alphabets and " " (space)
    filter = string.ascii_letters + " "
    text = "".join([chr for chr in text if chr in filter])                                                      # remove all characters except alphabets and " " (space)
    
    # Removing words with one alphabet occuring more than 3 times continuously
    pattern = r'\b\w*?(.)\1{2,}\w*\b'
    text = re.sub(pattern, "", text).strip()                                                                    # remove words with one alphabet occuring more than 3 times continuously
    
    # Removing words with less than 3 characters
    short_words = r'\b\w{1,2}\b'
    text = re.sub(short_words, "", text)                                                                     # remove words with less than 3 characters
    
    # return final output
    return text

# ===============================================================================================================
                                       # STREAMLIT

# App Devolopment Starts
st.set_page_config(layout="wide")
st.write("# A Predictive Analysis of Disaster Tweets")

img = Image.open("t2.png")
st.image(img)

tweet = st.text_input(label = "Type or paste your tweet here", value = "")

# Defining a function to store the model in streamlit cache memory
@st.cache_resource
def cache_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model

model = cache_model("transfer_tweet")

# if user gives any input
if len(tweet) > 0:
    clean_tweet = preprocess(tweet)                   # cleans tweet
    y_pred = model.predict([clean_tweet])             # gives probability of class = 1
    y_pred_num = int(np.round(y_pred)[0][0])          # get final prediction of output class
    
    if y_pred_num == 0:
        # st.write(f"#### Non-Disaster tweet with disaster probability {round(y_pred[0][0]*100, 4)}%")
        st.write(f"#### 🌞🌞This tweet is not flagged as a disaster, but with a probability of {round(y_pred[0][0]*100, 4)}% that it might be. ")
    else:
        st.write(f"#### 🚩🚩High probability ( {round(y_pred[0][0]*100, 4)}%) indicates that this tweet is related to a disaster🚨🚨.")

# ==============================================================================================================
        
# ---------------------------- Disaster Tweets -------------------------------
# "🚨 Just felt a strong earthquake! Stay safe everyone! #earthquake #safetyfirst"
# "⚠️ Urgent: Massive wildfire approaching our community. Evacuation orders in effect. Please heed warnings and evacuate immediately. #wildfire #safety"
# "🌪️ Tornado warning in effect for our area. Take shelter now! #tornadowarning #safetyfirst"
# "🌊 Coastal areas under tsunami alert. Seek higher ground immediately! #tsunami #emergencyalert"
# "⛈️ Severe thunderstorm approaching. Secure outdoor items and stay indoors. #thunderstorm #safety"
        

# ---------------------------- Non disaster Tweets -------------------------------
# "Enjoying a peaceful evening with a good book and a cup of tea. #Relaxation"
# "Excited for the weekend! Planning a movie night with friends. 🍿🎬 #FridayFeeling"
# "Just finished a great workout session at the gym. Feeling energized! 💪 #FitnessGoals"
# "Spent the day exploring a new hiking trail. Nature is so beautiful! 🌳 #OutdoorAdventure"
# "Cooked a delicious homemade dinner tonight. #Foodie #HomeChef"