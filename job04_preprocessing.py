import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# +konlpy 설치
from konlpy.tag import Okt
from tensorflow.keras.preprocessing. text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width',True)
df=pd.read_csv('./crawling_data/naver_news_titles_20220526.csv')
print(df.head())
df.info()

x=df['titles']
y=df['category']

encoder=LabelEncoder()
labeled_Y=encoder.fit_transform(y)#오름차순으로 정렬됨
#print(labeled_Y[:3])
label=encoder.classes_
#print(label)
with open('./models/encoder.pickle','wb')as f:
    pickle.dump(encoder,f)
onehot_Y=to_categorical(labeled_Y)
print(onehot_Y)

okt=Okt() #java oracle8 필요
# okt_morph_x=okt.morphs(x[7],stem=True) #morphs 형태소로 잘라줌
# print(okt_morph_x)

for i in range(len(x)):
    x[i]=okt.morphs(x[i],stem=True)
#print(x[:10])

stopwords=pd.read_csv('./crawling_data/stopwords.csv',index_col=0)

#불용어, 한글자 말 제거
for j in range(len(x)):
    words = []
    for i in range(len(x[j])):
        if len(x[j][i])>1:
            if x[j][i] not in list(stopwords['stopword']):
                words.append(x[j][i])
    x[j]=' '.join(words)
# print(x[:5])
# print(words)

#단어를 숫자로 토크나이징
token=Tokenizer()
token.fit_on_texts(x)
tokened_x=token.texts_to_sequences(x)
wordsize=len(token.word_index)+1 #단어 길이
# print(tokened_x)

print(token.word_index)
with open('./models/news_token.pickle','wb')as f:
    pickle.dump(token,f)

maximum=0
for i in range(len(tokened_x)):
    if maximum < len(tokened_x[i]):
        maximum=len(tokened_x[i])
print(maximum)

#maximum값에 맞춰 패딩
x_pad=pad_sequences(tokened_x,maximum)
print(x_pad)

#자연어 처리
x_train,x_test,y_train,y_test=train_test_split(
    x_pad,onehot_Y,test_size=0.1)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

xy=x_train,x_test,y_train,y_test
np.save('./crawling_data/news_data_max_{}_wordsize_{}'.format(maximum, wordsize), xy)
