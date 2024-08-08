import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
from technical_indicators import ema, bollinger_bands, rsi, macd, kama, ppo, atr, aroon, ichimoku

class Company:
    def __init__(self, ticker):
        self.ticker = ticker
        self.base_url = f"https://www.cophieu68.vn/quote/history.php?cP={{page}}&id={self.ticker}"
        self.english_headers = [
            'Date', 'CLOSE', 'VOLUME', 'OPEN', 'HIGH', 'LOW', 
            'Foreign Buy', 'Foreign Sell', 'Foreign Value\n(Billion VND)'
        ]
        self.stock_price = self.get_stock_price()
    
    def fetch_data(self, page):
        url = self.base_url.format(page=page)
        response = requests.get(url)
        response.raise_for_status()
        # print(f"Fetching data from: {url}")

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'history'})

        rows = []
        if table:
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) == len(self.english_headers):
                    rows.append([cell.text.strip().replace(',', '') for cell in cells])
        else:
            print(f"No table found on page {page}")
        return rows

    def get_stock_price(self):
        all_data = []
        page = 1
        while True:
            data = self.fetch_data(page)
            if not data:
                break
            all_data.extend(data)
            page += 1

        if not all_data:
            # print("No data fetched.")
            return pd.DataFrame(columns=self.english_headers)

        df = pd.DataFrame(all_data, columns=self.english_headers)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df.set_index('Date', inplace=True)
        
        # Convert columns to numeric, ignoring errors for non-numeric data
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return self.calculate_features(df)

    def calculate_features(self, df):
        df['Avg20_Vol'] = df['VOLUME'].rolling(20).mean()
        df['EMA5'] = ema(df['CLOSE'], span=5)
        df['EMA50'] = ema(df['CLOSE'], span=50)
        df['High_BB'], df['Low_BB'], df['SMA20'] = bollinger_bands(df['CLOSE'], window=20, window_dev=2)
        df['RSI'] = rsi(df['CLOSE'], window=14)
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = macd(df['CLOSE'], slow=26, fast=12, signal=9)
        df['KAMA'] = kama(df['CLOSE'], span=10)
        df['PPO'], df['PPO_SIGNAL'] = ppo(df['CLOSE'], slow=26, fast=12, signal=9)
        df['ATR'] = atr(df['HIGH'], df['LOW'], df['CLOSE'], window=14)
        df['AROON_UP'], df['AROON_DOWN'], df['AROON_INDICATOR'] = aroon(df['CLOSE'], window=25)
        df['TENKAN_SEN'], df['KIJUN_SEN'], df['SENKOU_SPAN_A'], df['SENKOU_SPAN_B'], df['CHIKOU_SPAN'] = ichimoku(df['HIGH'], df['LOW'], df['CLOSE'])
        df['DAILY_RET(T)'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))

        df = df.dropna()
        df.columns = [col + '_' + self.ticker for col in df.columns]
        df.columns = [col.upper() for col in df.columns]
        df = df[df.index >= datetime.strptime('2010-01-01', '%Y-%m-%d')]
        
        return df

