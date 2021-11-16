import re
import nltk
import pandas as pd
import numpy as np
dataset=pd.read_csv("20191226-reviews.csv")
print(dataset.head())
print(dataset.isnull().sum())
dataset['body']=dataset ['body'].fillna('').apply(str)

dataset['name'] = dataset ['name'].fillna('').apply(str) 
dataset['title'] = dataset ['title'].fillna('').apply(str) 
dataset['helpfulVotes'] = dataset [ 'helpfulVotes' ].fillna('').apply(str)
print(dataset.isnull().sum())
dataset=dataset.drop(columns=['asin','name','helpfulVotes','date'],axis=1)
a=dataset['rating'].tolist()
d=[]
for i in range(len(a)):
    if a[i]>=3:
        d.append(1)
    else:
        d.append(0)
print(d)        
dt=pd.DataFrame(d,columns=['emotion'])
print(dt)
data1=pd.concat([dataset,dt],axis=1)
data1.head()
data1.drop(['verified'],axis=1,inplace=True)
data1['Review'] = data1['title'].str.cat(data1['body'],sep=" ")
data1.drop(['title','body','rating'],axis=1,inplace=True)
print(data1.head())
print(data1.shape)
y=data1.iloc[:,0].values
x=data1.iloc[:,1].values
print(y)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #create an object for stemming
ps=PorterStemmer()
#library used for stem the words
from nltk. stem import WordNetLemmatizer #create an object for wordnet Lemmatizer 
wordnet=WordNetLemmatizer()
data=[]
for i in range(len(x)):
    review=data1['Review'][i]
    review=re.sub('[^a-zA-Z]',' ',str(review))
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    data.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(data).toarray()
print(x)
import pickle
pickle.dump(cv,open('count_vec.pkl','wb'))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(units=13264,activation ='relu'))
model.add(Dense(units= 2000,activation ='relu'))
model.add(Dense(units= 2000,activation ='relu'))
model.add(Dense(units= 2000,activation ='relu'))
model.add(Dense(units=1,activation ='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=50)
y_pred = model.predict(x_test)
text = "The phone is okay. average " 
text = re.sub ('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)
y_p = model.predict(cv.transform( [text]))
# saving the model
model.save("review_analysis.h5") 
