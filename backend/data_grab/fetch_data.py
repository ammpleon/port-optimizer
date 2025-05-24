import pandas as pd
import numpy as np
import yfinance as yf  
import datetime as dt
import requests as r
from dotenv import load_dotenv
import os



load_dotenv()

class DataFetcher:
   """This class fetches ticker data from yahoo finance"""
   def __init__(self, endDate, years, tickers):
      self.tickers = tickers if isinstance(tickers, list) else [tickers]
      self.years = years
      self.endDate = endDate
      self.startDate = endDate - dt.timedelta(days = 365 * float(self.years))


   def get_start_date(self):
      """Only returns the start date"""
      return self.startDate


   def validate_tickers(self):
      
      """return valid tickers if they are valid in yahoo finance"""

      valid_tickers = []
      for ticker in self.tickers:
         data = yf.Ticker(ticker).history('1d')
         if not data.empty:
            valid_tickers.append(ticker)
    
      return valid_tickers

   def fetch_data(self):
      """call the validate_tickers() method to ensure that all tickers are valid then fetch the data"""

      valid_tickers = self.validate_tickers()

      if not valid_tickers:
         return pd.DataFrame()
      
      df = pd.DataFrame()

      for ticker in self.tickers:
         data = yf.download(start = self.startDate, end = self.endDate, tickers = ticker, auto_adjust= False)
         if data.empty:
            continue
         else:

            if "Adj Close" in data.columns:
               df[ticker] = data["Adj Close"]
            else:
               df[ticker] = data["Close"]



      # #forward fill the NaN values of the valid tickers
      # df.fillna(method = 'ffill', inplace = True)

      

      #remove the columns that contains no data and is empty.
      df = df.dropna()

      return df 
   
   @staticmethod
   def _get_api_key():
      """Retrieve the API key securely from environment variables."""
      api_key = os.getenv("TREASURY_API_KEY")
      if not api_key:
         raise ValueError("API key is missing. set TREASURY_API_KEY in environment variables.")
      return api_key



   @staticmethod
   def _fetch_treasury():
      """Fetches the treasury rate of the stock. Does not if the ticker is crypto/currency-pair"""
      api_key = DataFetcher._get_api_key()
      params = {"api_key": api_key,
             "file_type": "json",
             "series_id": "DGS10",
             "observation_start": (dt.datetime.now() - dt.timedelta(days = 365* 20)).strftime(f"%Y-%m-%d"),
             "observation_end": dt.datetime.now().strftime(f"%Y-%m-%d"),}
      
      FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"  
      response = r.get(FRED_ENDPOINT, params = params)

      data = response.json()  

      rf_df = pd.DataFrame(data["observations"])
      rf_df["date"] = pd.to_datetime(rf_df["date"])
      rf_df["value"] = rf_df["value"]

      rf_df = rf_df.drop(["realtime_start", "realtime_end"], axis = 1)
      rf_df["value"] = pd.to_numeric(rf_df["value"], errors = "coerce")

      rf_df = rf_df.rename({'date': 'Date'}, axis = 1)

      rf_df = rf_df.set_index("Date")



      return rf_df


   # def fetch_sukuk():
   #    sukuk_df = pd.read_csv('https://www.econdb.com/api/series/Y10YDSA/?format=csv&token=f5fb6396efc8d3548202daf54b092c43cf78e4fa',
	#    index_col='Date', parse_dates=['Date'])

   #    return sukuk_df






   


