from cProfile import label
from json import load
from pyexpat import model
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


def past_date(selected):
    match selected:
        case "1-Year":
            return date.today() + relativedelta(years=-1)
        case "2-Year":
            return date.today() + relativedelta(years=-2)
        case "3-Year":
            return date.today() + relativedelta(years=-3)
        case "5-Year":
            return date.today() + relativedelta(years=-5)
        case "10-Year":
            return date.today() + relativedelta(years=-10)
        case "20-Year":
            return date.today() + relativedelta(years=-20)
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker' , 'SBIN.NS')
date_range = ["1-Year","2-Year","3-Year","5-Year","10-Year","20-Year"]
date_range = st.selectbox("Select Time Period", options= date_range)
selected = date_range
start = past_date(selected)
end = date.today()
df = data.DataReader(user_input,'yahoo',start ,end)

#Describing Data
st.write('Data Form ',start,'-',end)
st.write(df.describe())

#Data Visualizations
st.subheader('Closing Price vs Time-Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close )
st.pyplot(fig)

st.subheader('Closing Price vs Time-Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(df.Close,'b',label='Closing')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time-Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='MA100')
plt.plot(ma200,'g',label='MA200')
plt.plot(df.Close,'b',label='Closing')
plt.legend()
st.pyplot(fig)

#Splitting Data Into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Load My model
model=load_model('Keras_Model.h5')

# Testing Part
last_100_days = data_training.tail(100)
final_Df = last_100_days.append(data_testing , ignore_index=True)
input_data = scaler.fit_transform(final_Df)

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