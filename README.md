# Stock_Price_Prediction_on_Future


This code is an example of time series analysis. 
Using time series analysis we can fill missing values in a series or we can approximate past or future data. 

This model is created using Dense Neural Network (DNN), Recurrent Neural Network (RNN), Long Short Term Memory (LSTM). But I found DNN most efficient. Somehow I also created RNN which is uploaded in this repository. I also created LSTM for the same data but LSTM took much more time than DNN. So I didn't upload it.

Data used in this model is Tata Consultancy Services (TCS.NS) data from 12th August 2002.
In the model-accuracy csv file I measured the performance of this model. 

The previous 100 days data will be the feature values and next day's data will be the output value for this model.
