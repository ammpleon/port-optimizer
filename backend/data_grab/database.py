import psycopg2
from dotenv import load_dotenv
import os
import yfinance as yf
from backend.data_grab import DataFetcher
import datetime as dt
load_dotenv()




conn = psycopg2.connect(host = os.getenv("host"),
                        password = os.getenv("password"),
                        port = os.getenv("port"),
                        user = os.getenv("user"),
                        dbname = "portfolio_optimizer")




cur = conn.cursor()



assets2 = ["7010.SR", "1180.SR",
           "2222.SR", "1010.SR",
          "1150.SR", "TSLA",
          "MSFT", "NVDA",
          "AMD", "6033.KL",
          "BTC-USD", "SPY",
          "VGT", "ETH-USD",
          "FBMPM.FGI", "GC=F",
          "^TASI.SR", "^GSPC",
          "^N225", "XRP-USD",
          "HOOD", "INTC",
          "AMZN", "GOOG",
          "WOLF", "META",
          "UBER", "BNB-USD",
          "DOGE-USD", "USDT-USD",
          "SOL-USD", "SHIB-USD",
          "2223.SR"  
          ]



#storing the asset's meta data
tick_info = {}

for asset in assets2:
    try:
        ticker_data = yf.Ticker(ticker = asset).info
    except AttributeError:
        continue
    name = ticker_data.get("shortName")
    asset_type = ticker_data.get("typeDisp").lower()
    exchange = ticker_data.get("fullExchangeName")

    sector_map = {
        "cryptocurrency": "Crypto",
        "etf" : "Broad Market",
        "futures" : "Commodity",
        "index" : "Index"
    }

    sector = sector_map.get(asset_type, ticker_data.get("sector", "Unknown"))
    tick_info[asset] = (name, asset_type, sector, exchange)


for ticker, (name, type, sector, exchange) in tick_info.items():
    cur.execute('INSERT INTO asset_metadata VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING ',
                (ticker, name, type, sector, exchange))


#Commit and close

conn.commit()
cur.close()
conn.close()