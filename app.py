from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

model = pickle.load(open('flight_rf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit',methods=['POST'])
def results():
    key = [i for i in request.form.keys()]
    value = [i for i in request.form.values()]
    df = pd.DataFrame([value],columns=key)
    df['Airline']=pd.Categorical(df['Airline'],categories=['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'Go Air', 'Vistara', 'Air Asia', 'Vistara Premium economy', 'Jet Airways Business', 'Multiple carriers Premium economy', 'Trujet'])
    df['Source']=pd.Categorical(df['Source'],categories=['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    df['Destination']=pd.Categorical(df['Destination'],categories=['New Delhi', 'Bangalore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    df=pd.get_dummies(df,columns=['Airline','Source','Destination'],drop_first=True)
    prediction= model.predict(df)
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text= 'The Estimated Price for your flight is Rs {}'.format(output))
if __name__ == '__main__':
    app.run()
