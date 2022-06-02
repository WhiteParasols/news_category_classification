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
from tensorflow.keras.models import load_model

pd.set_option('display.unicode.east_asian_width',True)
pd.set_option('display.max_columns',20)
df=pd.read_csv('./crawling_data/naver_headline_news20220527.csv')
print(df.head())
df.info()

x=df['titles']
y=df['category']

with open('./models/encoder.pickle','rb')as f:
    encoder=pickle.load(f)

labeled_Y=encoder.transform(y)#오름차순으로 정렬됨
#print(labeled_Y[:3])
label=encoder.classes_
#print(label)

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

with open('./models/news_token.pickle','rb') as f:
    token=pickle.load(f)

tokened_x=token.texts_to_sequences(x)
for i in range(len(tokened_x)):
    if len(tokened_x[i])>17:
        tokened_x[i]=tokened_x[i][:17]

x_pad=pad_sequences(tokened_x,17)
print((x_pad[:5]))

model=load_model('./models/news_category_classification_model_0.6700336933135986.h5')
preds=model.predict(x_pad)
predicts=[]
for pred in preds:
    most=label[np.argmax(pred)]
    pred[np.argmax(pred)]=0
    second=label[np.argmax(pred)]
    predicts.append([most,second])
df['predict']=predicts
print(df.head(30))
df['OX']=0
for i in range(len(df)):
    if df.loc[i,'category']in df.loc[i,'predict']:
        df.loc[i,'OX']='O'
    else:
        df.loc[i,'OX']='X'
print(df.head(30))
print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))

for i in range(len(df)):
    if df['category'][i] not in df['predict'][i]:
        print(df.iloc[i])