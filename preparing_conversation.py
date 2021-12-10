from nltk.util import pr
import pandas as pd
import pickle
from pyvi import ViTokenizer, ViPosTagger
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj    

def isNaN(string):
    return string != string

stop_words = set(stopwords.words('FB_vietnamese')) 
def standardize_data(row):
    row = row.strip().lower()
    word_tokens = word_tokenize(row) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    text = " ".join(filtered_sentence)
    text = text.replace(" . . ", ".").replace(" .", ".")\
        .replace(":", " ").replace("!", " ")\
        .replace('"', " ").replace("'", " ").replace("..", ".")\
        .replace(". ."," ").replace(" . ", " ").replace(", "," ").replace("  ", " ") 
    text=text.replace(" .",".").replace("  ", " ") 
    return ViTokenizer.tokenize(text)
def text_cleaning(lines):
    for i in range (len(lines)):
        line=standardize_data(lines[i])
        #filter sentences with number of words >3
        if(not isNaN(line) and len(line.split())>3):
            clean_train_data.append(line)
#parameters
ID_NTU=1181954081822765
start_month=4
end_month=9
train_data = []
clean_train_data=[]   
line="" 
conversation=0
trainingdata=pd.read_csv('data/raw_FB_dataset.csv')
for i in range(len(trainingdata)):
    if not isNaN(trainingdata['message'][i]) \
        and (trainingdata['message'][i].lower() not in stop_words)\
        and trainingdata['from_id'][i]!=ID_NTU\
        and (int(trainingdata['time'][i][5:7])>=start_month)\
        and (int(trainingdata['time'][i][5:7])<=end_month):
        line=trainingdata['message'][i]
        conversation +=1
        train_data.append(line)
        if(conversation%50==0): 
            print("done processing raw line: ", conversation)                       
text_cleaning(train_data)
# print(clean_train_data)
_save_pkl('DLU_text_train.pkl', clean_train_data)
print("number of conservation: ", len(clean_train_data))