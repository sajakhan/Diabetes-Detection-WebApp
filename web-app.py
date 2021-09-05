from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder

st.write(""" Diabetes Detection App""")
df=pd.read_csv(r"C:\Users\Hp\Desktop\diabetes-detection-webapp\diabetes.csv")
st.subheader('Data Information: ')
st.dataframe(df)
st.write(df.describe())
chart=st.bar_chart(df)

x= df.iloc[:,0:8].values
y=df.iloc[:,-1].values
#75%train 25% test
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)

def get_user_input():
    pregnancies=st.sidebar.slider('pregnancies',0,17,3)
    glucose=st.sidebar.slider('glucose',0,199,117)
    blood_pressure= st.sidebar.slider('blood_pressure',0,122,72)
    skin_thickness= st.sidebar.slider('skin_thickness',0,99,3)
    insulin= st.sidebar.slider('insulin',0.0,846.0,30.5)
    BMI= st.sidebar.slider('BMI',0.0,67.1,32.1)
    DPF= st.sidebar.slider('DPF',0.078,2.42,0.375)
    age= st.sidebar.slider('age',21,81,29)
    
    #store dictionary into a varaiable
    user_data={
        'pregnancies':pregnancies,
        'glucose':glucose,
        'blood_pressure':blood_pressure,
        'skin_thickness':skin_thickness,
        'insulin':insulin,
        'BMI':BMI,
        'DPF':DPF,
        'age':age

    }
    #transform the data into a data frame
    features=pd.DataFrame(user_data,index={0})
    return features


user_input = get_user_input()
st.subheader('user input:')
st.write(user_input)

RFC= RandomForestClassifier()
RFC.fit(x_train,y_train)
pred=RFC.predict(x_test)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test,pred)*100)+'%')



#store model predicition in variable
prediction=RFC.predict(user_input)
st.subheader('Classification: ')
if(prediction==1):

	st.write('Diabetic')

if(prediction==0):

	st.write('Non - Diabetic')

st.write(prediction)