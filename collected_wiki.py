from nltk.util import pr
import pandas as pd
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from pyvi import ViTokenizer
import pandas as pd
import wikipedia
wikipedia.set_lang("vi")
def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj  
articles=['phim kinh dị', 'nhạc trữ tình', 'Giải bóng đá Ngoại hạng Anh', 'Hội họa dân gian Việt Nam', 
            'Trí tuệ nhân tạo']
# articles=['Trí tuệ nhân tạo']            
wiki_lst=[]
title=[]
for article in articles:
    print("loading content: ",article)
    wiki_lst.append(wikipedia.page(article).content)
    title.append(article)
print("examine content")
# a_list = nltk.tokenize.sent_tokenize(wiki_lst[0])
clean_text=[]
label=[]
odering=0
def isNaN(string):
    return string != string
stop_words = set(stopwords.words('FB_vietnamese')) 
for paragraph in wiki_lst:
    a_list = nltk.tokenize.sent_tokenize(paragraph)
    for sentences in a_list:
        sentence=" ".join(sentences.split())#remove whitespace
        if(sentence[0:2]!='==' or isNaN(sentence)): 
            word_tokens = word_tokenize(sentence)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            text = " ".join(filtered_sentence)
            clean_text.append(ViTokenizer.tokenize(text))
            label.append(odering)
    odering=odering+1
print(len((clean_text)), len(label))
#write file 
dirName = 'data'   
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")
train_textfile = open("data/train_wiki.txt", "w")
for element in clean_text:
    train_textfile.write(element + "\n")
train_textfile.close()
train_textfile = open("data/label_wiki.txt", "w")
for element in label:
    train_textfile.write(str(element) + "\n")
train_textfile.close()
_save_pkl('data/train_wiki.pkl', clean_text)
_save_pkl('data/label_wik.pkl', label)