import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from technical_indicators import ema, bollinger_bands, rsi, macd, kama, ppo, atr, aroon, ichimoku

pd.set_option('display.max_columns', 999)

class Company:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock_price = self.get_stock_price()

    def get_stock_price(self):
        close_price_arr = []
        open_price_arr = []
        high_price_arr = []
        low_price_arr = []
        volume_arr = []
        date_arr = []

        url = 'http://www.cophieu68.vn/historyprice.php'
        for page in range(1, 40):
            params = {'currentPage': str(page), 'id': self.ticker}
            result = requests.get(url, params=params)
            doc = BeautifulSoup(result.text, 'html.parser')

            total_price_arr = doc.find_all(['span'], class_=['priceup', 'pricedown'])
            date_volume_arr = doc.find_all(class_='td_bottom3 td_bg1')

            for j in range(0, int(len(total_price_arr) / 6)):
                close_price_arr.append(total_price_arr[2 + j * 6])
                open_price_arr.append(total_price_arr[3 + j * 6])
                high_price_arr.append(total_price_arr[4 + j * 6])
                low_price_arr.append(total_price_arr[5 + j * 6])

            for j in range(0, int(len(date_volume_arr) / 7)):
                date_arr.append(date_volume_arr[1 + j * 7])
                volume_arr.append(date_volume_arr[3 + j * 7])

        close_price_arr = [float(price.string.replace(',', '')) for price in close_price_arr]
        open_price_arr = [float(price.string.replace(',', '')) for price in open_price_arr]
        high_price_arr = [float(price.string.replace(',', '')) for price in high_price_arr]
        low_price_arr = [float(price.string.replace(',', '')) for price in low_price_arr]
        date_arr = [datetime.strptime(date.string, '%d-%m-%Y') for date in date_arr]
        volume_arr = [float(volume.string.replace(',', '')) for volume in volume_arr]

        stock_price = pd.DataFrame(index=date_arr)
        stock_price['Open'] = open_price_arr
        stock_price['High'] = high_price_arr
        stock_price['Low'] = low_price_arr
        stock_price['Close'] = close_price_arr
        stock_price['Volume'] = volume_arr

        stock_price = stock_price.iloc[::-1]
        stock_price['Avg20_Vol'] = stock_price['Volume'].rolling(20).mean()

        stock_price['EMA5'] = ema(stock_price['Close'], span=5)
        stock_price['EMA50'] = ema(stock_price['Close'], span=50)

        stock_price['High_BB'], stock_price['Low_BB'], stock_price['SMA20'] = bollinger_bands(stock_price['Close'], window=20, window_dev=2)

        stock_price['RSI'] = rsi(stock_price['Close'], window=14)

        stock_price['MACD'], stock_price['MACD_signal'], stock_price['MACD_hist'] = macd(stock_price['Close'], slow=26, fast=12, signal=9)

        stock_price['KAMA'] = kama(stock_price['Close'], span=10)

        stock_price['PPO'], stock_price['PPO_signal'] = ppo(stock_price['Close'], slow=26, fast=12, signal=9)

        stock_price['ATR'] = atr(stock_price['High'], stock_price['Low'], stock_price['Close'], window=14)

        stock_price['Aroon_up'], stock_price['Aroon_down'], stock_price['Aroon_indicator'] = aroon(stock_price['Close'], window=25)

        stock_price['Tenkan_sen'], stock_price['Kijun_sen'], stock_price['Senkou_span_a'], stock_price['Senkou_span_b'], stock_price['Chikou_span'] = ichimoku(stock_price['High'], stock_price['Low'], stock_price['Close'])

        stock_price['Daily_ret(t)'] = np.log(stock_price['Close'] / stock_price['Close'].shift(1))

        stock_price = stock_price.dropna()
        stock_price.columns = [col + '_' + self.ticker for col in stock_price.columns]
        stock_price.columns = [col.upper() for col in stock_price.columns]
        stock_price = stock_price[stock_price.index >= datetime.strptime('2010-01-01', '%Y-%m-%d')]

        return stock_price
