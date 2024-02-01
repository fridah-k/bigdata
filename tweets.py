import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer

df= pd.read_csv('/home/fridah/Downloads/tweets.csv')
print(df.head())
txt_data=df.loc[:,'text']
print(txt_data)
# Transform column Question 3
df_new = df.rename(columns={'id': 'paroles'}, index={'1': 'izooh'})
print(df_new)
#question 4. Using Re
for lyrics in txt_data:
    re.search('()',lyrics)
    lyrics.replace('()'," ")
# print(lyrics)
# myname = "oyaaah (mapenzi) mzae (katoto) radaa"
lyrics=re.findall(r'\((.*?)\)',lyrics)
print("**********************************")
print(lyrics)
print("**********************************")

# myname.replace("()"," ...")




# #Question 5 Stopword
# print(name)
# stop_words= set(stopwords.words('english'))
# print(stop_words)

# # Question no six. adding character strings

# res= "?", "!", ".", "", ":","they've", "they're","they'll", "i've", "i'm", "i'll", "could,".join(stop_words)
# print( res)

# # Question 7.
# # from nltk.tokenize import TweetTokenizer
# # print(TweetTokenizer(stop_words))

# # Question 8
# tweet_tokenizer = TweetTokenizer()
# tweet_tokens = []
# for words in stop_words:
#     print(tweet_tokenizer.tokenize(words))
#     tweet_tokens.append(tweet_tokenizer.tokenize(words))

# # Question 9
# # token_lyrics = PunktSentenceTokenizer(lyrics)

# # print("\nSentence-tokenized copy in a list:")
# # for mots in token_lyrics:
# #     print(mots)





