import pandas_datareader.data as pdr
import yfinance as yfin

# Override pdr due to lack of connection between pdr and yahoo
yfin.pdr_override() 

class df:

    def __init__(self):

        self._data = None
        self._stock = None
        self._start = None
        self._end = None

        stock_valid = False

        while not stock_valid:
            stock = input("Enter Stock Name: ")
            print("Enter Start Date: ")
            start = self.get_date()
            print("Enter Start Date: ")
            end = self.get_date()

            stock_valid = self.check_stock_valid(stock, start, end)
        
        self._start = start
        self._end = end

    def check_stock_valid(self, stock_name, start, end):

        try:
            self._data = pdr.get_data_yahoo(stock_name, start= start, end= end)
            self._stock = stock_name
            return True
        
        except:
            print("Error: Stock Data Not Available.")
            return False
    
    def get_date(self):

        day = input("Enter the day: ")
        month = input("Enter the month: ")
        year = input("Enter the year: ")

        date = str(year) + "-" + str(month) + "-" + str(day)
        print(date)

        return date
    
    def get_data(self):
        return self._data
    
    def get_stock(self):
        return self._stock
    
    def get_start(self):
        return self._start
    
    def get_end(self):
        return self._end