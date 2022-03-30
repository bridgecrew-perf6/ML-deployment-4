from flask import Flask,render_template,request
import pickle
import numpy as np 
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    features=[int(x) for x in request.form.values()]
    new_features=np.array(features).reshape(1,3)
    prediction=model.predict(new_features)
    prediction_values={0:"Not Purchased",1:"Purchased"}
    result=prediction_values[prediction[0]]
    return render_template('result.html',prediction_text="You Have {}".format(result))

if __name__=="__main__":
    app.run(port=8000)