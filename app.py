
from flask import render_template, Flask, request,url_for
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
graph =tf.compat.v1.get_default_graph()
with open(r'count_vec.pkl','rb') as file:
    cv=pickle.load(file)
app=Flask(__name__)
@app.route('/') 
def home():
    return render_template('index.html')
@app.route('/tpredict')

@app.route('/', methods=['GET', 'POST'])

def page2():
    if request.method == 'GET':

      return render_template('index.html')

    if request.method == 'POST' : 
       topic= request.form['tweet'] 
       print("Hey " +topic) 
       topic=cv.transform([topic]) 
       print("\n"+str(topic.shape)+"\n")
       with graph.as_default():
           cla = load_model('review_analysis.h5')
           cla.compile(optimizer='adam',loss='binary_crossentropy')
           y_pred = cla.predict(topic) 
           print("pred is "+str(y_pred))

       if(y_pred > 0.7):
           topic="Positive Review"

       else:
         topic = "Negative Review"
       return render_template('index.html', ypred = topic)

if __name__=="__main__":
    app.run(debug=False)