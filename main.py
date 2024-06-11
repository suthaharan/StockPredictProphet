# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import pandas as pd

START = "2023-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.set_page_config('Moving average CandleStick Chart', layout='wide')
st.sidebar.header("Select Ticker and Date")


# stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
# selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
selected_stock=st.sidebar.text_input('Ticker', 'AAPL')
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# today = date.today().strftime("%Y-%m-%d")
ticker = selected_stock
start_date = st.sidebar.text_input('Start date', f'{START}')
end_date = st.sidebar.text_input('End date', f'{TODAY}')
st.title('Stock Forecast ' + f'{ticker}' + " - " + f'{start_date}')

#@st.cache_data 
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], line=dict(color='green', width=1), name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], line=dict(color='red', width=1), name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


st.header("Moving average and Candlestick chart of "+ ticker)
# # st.set_page_config('Moving average CandleStick Chart', layout='wide')
# # st.sidebar.header("Select Ticker and Date")
# # today = date.today().strftime("%Y-%m-%d")

# # ticker=st.sidebar.text_input('Ticker', 'AAPL')
# # st.header("Moving average and Candlestick chart of "+ ticker)

# start_date = st.sidebar.text_input('Start date', f'{START}')
# end_date = st.sidebar.text_input('End date', f'{TODAY}')

average1 = st.sidebar.number_input('Insert a moving average', value=200)
average2 = st.sidebar.number_input('Insert a moving average', value=400)

start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date)

# data=yf.download(ticker, start, end)

st.dataframe(data, width=1800, height=600)
df = pd.DataFrame(data)
df['MA1'] = df.Close.rolling(average1).mean()
df['MA2'] = df.Close.rolling(average2).mean()

fig3 = go.Figure(data=[go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name='Candle Stick'),
                       go.Scatter(x=df.index, y=df.MA1, line=dict(color='orange', width=1), name='Moving Avg '+ f'{average1}'),
                       go.Scatter(x=df.index, y=df.MA2, line=dict(color='green', width=1), name='Moving Avg '+ f'{average2}'),
                       ])

fig3.update_layout(autosize=False, width=1400, height=800)
st.plotly_chart(fig3)