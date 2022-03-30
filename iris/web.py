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
    iris_features=[float(x) for x in request.form.values()]
    entered_features=np.array(iris_features).reshape(1,4)
    prediction=model.predict(entered_features)
    prediction_values={0:"Iris-setosa" ,1:"Iris-versicolor" ,2:"Iris-virginica"}
    result=prediction_values[prediction[0]]
    return render_template('result.html',prediction_text="It belongs to {}".format(result))


if __name__=='__main__':
    app.run(port=8000)