from bs4 import BeautifulSoup
import requests as re
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin


def scrape_request(func):
    def wrapper(url, headers, *args, **kwargs):
        try:
            response = re.get(url, headers = headers)
            if response.status_code == 200:
                print(f"Access Granted. ({response.status_code})")
                return func(response, url, headers, *args, **kwargs)
            else:
                print(f"Access Failed. ({response.status_code})")

        except re.exceptions.RequestException as e:
            print(f"Request failed {e}")
            return []
    return wrapper




url = "https://www.econdb.com/series/Y10YDSA/saudi-arabia-long-term-yield/?from=2020-03-29&to=2025-03-29"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://google.com"
}



div_list = []
response = re.get(url = url, headers = headers)

soup = BeautifulSoup(response.content, "html.parser")

div = soup.find("div")

import pandas as pd
df = pd.read_csv(
	'https://www.econdb.com/api/series/Y10YDSA/?format=csv&token=f5fb6396efc8d3548202daf54b092c43cf78e4fa',
	index_col='Date', parse_dates=['Date'])


print(df)





# print(div_list)

