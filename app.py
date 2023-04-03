# Importing necessary files
from cProfile import label
from json import load
from pyexpat import model
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st


# creating a function to get starting date from the selected date range using selectbox
def past_date(selected):
    switcher = {
        "1-Year":
            date.today() + relativedelta(years=-1),
        "2-Year":
            date.today() + relativedelta(years=-2),
        "3-Year":
            date.today() + relativedelta(years=-3),
        "5-Year":
            date.today() + relativedelta(years=-5),
        "10-Year":
            date.today() + relativedelta(years=-10),
        "20-Year":
            date.today() + relativedelta(years=-20),
    }
    return switcher.get(selected, date.today())


# Using streamlit to deploy the app and creating a selectbox
st.title('Stock Trend Prediction')

# Taking input for the stock ticker and the date range from the user  
user_input = st.text_input('Enter Stock Ticker' , 'SBIN.NS')
date_range = ["10-Year","1-Year","2-Year","3-Year","5-Year","20-Year"]
date_range = st.selectbox("Select Time Period", options= date_range)
selected = date_range
start = past_date(selected)
end = date.today()
data = yf.download(user_input, start=start, end=end)

#Describing Data
st.write('Data Form ',start,'-',end)
st.write(data.describe())

#Data Visualizations
st.subheader('Closing Price vs Time-Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close )
st.pyplot(fig)

st.subheader('Closing Price vs Time-Chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(data.Close,'b',label='Closing')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time-Chart with 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(ma200,'g',label='MA200')
plt.plot(data.Close,'b',label='Closing')
plt.legend()
st.pyplot(fig)

#Splitting Data Into Training and Testing

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Loading our keras model and predicting the values
model=load_model('Keras_Model.h5')

# Testing the model for the starting 100 days from the selected range
last_100_days = data_training.tail(100)
final_data = last_100_days.append(data_testing , ignore_index=True)
input_data = scaler.fit_transform(final_data)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test =np.array(x_test) , np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label = 'Original Price')
plt.plot(y_predicted , 'r' , label = ' Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
