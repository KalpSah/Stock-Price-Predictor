import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn import preprocessing
from tabulate import tabulate

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon=im
)

st.title('Stock Trend Prediction')

# input the Ticker
start_date = st.date_input('Enter the start Date: ',
                           datetime.datetime(2010, 5, 17))
end_date = st.date_input('Enter the End Date: ')

user_input = st.text_input('Enter the Stock Ticker', 'TTM')
st.write("[Find Your Ticker](https://stockanalysis.com/stocks/#:~:text=6067%20Stocks%20%20%20%20Symbol%20%20,%20%2039.85M%20%2036%20more%20rows%20)")
# df = web.DataReader(user_input, 'yahoo', start_date, end_date)
df = yf.download(user_input, start=start_date, end=end_date)


def stock_trend_prediction():
    
    
    #  Describing Data
    st.subheader('Data From {} - {}'.format(start_date.year, end_date.year))
    st.write(df.describe())
    st.subheader('Data last 5 days')
    st.write(df.tail())
   
   

    # Visualizations
    
    st.subheader('Closing Price vs Time Chart')
    b=st.line_chart(df.Close,height=250,width=1200)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    new_list=np.column_stack((ma100,df.Close,ma200))
    df1=pd.DataFrame(new_list,columns=['MA100','Original','MA200'])
    st.line_chart(df1,width =10,height=250,use_container_width=True)
   
    # Spliting Data into Training and testing
    data_training = pd.DataFrame(df['Close'][0: int(len(df) * 0.7)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Load my model
    model = load_model('model.h5')

    # Testing Part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    
    

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    
    
    
    y_predicted=scaler.inverse_transform(y_predicted) 
    y_test=y_test.reshape(1,-1)
    y_test=scaler.inverse_transform(y_test)
  

    # final Graph
    
    new_list2=np.column_stack((y_test[0],y_predicted))
    df2=pd.DataFrame(new_list2,columns=['Original Price','Predicted Price'])
    
    
    s='Predictions vs Original (last '+str(df2.shape[0])+' days)'
    st.subheader(s)
    st.line_chart(df2,height=250)
   
  
    
    web_link='[More Details](https://finance.yahoo.com/quote/'+user_input+')'
    st.markdown(web_link,  unsafe_allow_html=True)
    test=df2.tail()
    
    st.subheader('Predictions last 5 days')
    
    test= tabulate(test,headers='keys',tablefmt='github',showindex=False)
    st.write(test)

    
if(st.button('Prediction')):
    if(df.empty):
        st.error("ERROR!!! Check Stock Ticker")
    else:
        stock_trend_prediction()
