import pandas as pd
import numpy as np
import warnings
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from datetime import datetime, timedelta
import scipy.stats as st
import statsmodels.api as sm
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy import poly1d
warnings.simplefilter(action='ignore',  category=Warning)
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from pandas.tseries.offsets import BDay
from plotly.subplots import make_subplots
matplotlib.rcParams['figure.figsize'] = (25.0, 15.0)
matplotlib.style.use('ggplot')
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import plotly.io as pio
from numpy import median, mean

pio.templates.default = "plotly_white"

class SampleStrategy():
    
    def shortEntry(self, prices_df):
        
        short_entry_filter_1 = prices_df['MA NEAR'][-1] < prices_df['MA FAR'][-1]
        short_entry_filter_2 = prices_df['MA NEAR'][-2] > prices_df['MA FAR'][-2]

        enter_trade = short_entry_filter_1 and short_entry_filter_2

        if enter_trade:
            return True
        else:
            return False
    def longEntry(self, prices_df):
        
        long_entry_filter_1 = prices_df['MA NEAR'][-1] > prices_df['MA FAR'][-1]
        long_entry_filter_2 = prices_df['MA NEAR'][-2] < prices_df['MA FAR'][-2]

        enter_trade = long_entry_filter_1 and long_entry_filter_2

        if enter_trade:   
            return True
        else:
            return False
    
    def longExit(self, prices_df):
        
        long_exit_filter_1 = prices_df['MA NEAR'][-1] < prices_df['MA FAR'][-1]
        long_exit_filter_2 = prices_df['MA NEAR'][-2] > prices_df['MA FAR'][-2]
        
        exit_trade = long_exit_filter_1 and long_exit_filter_2

        if exit_trade:
                return True
        else:    
            return False
 
    def shortExit(self, prices_df):
        
        short_exit_filter_1 = prices_df['MA NEAR'][-1] > prices_df['MA FAR'][-1]
        short_exit_filter_2 = prices_df['MA NEAR'][-2] < prices_df['MA FAR'][-2]
        
        exit_trade = short_exit_filter_1 and short_exit_filter_2

        if exit_trade:
            return True
        else:    
            return False


from functools import reduce
class Broker():
    def __init__(self,
                 price_data=None,
                 MA_period_far=200,
                 MA_period_near=50):
        
        assert price_data is not None
        self.data = price_data
        # if mode == 'test':
        #     url='https://drive.google.com/file/d/15m4eQ1OYO8tNQ8grAS57FIjRFhNHcOGG/view?usp=sharing'
        #     url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
        #     self.data = pd.read_csv(url2 ,
        #                      parse_dates=['Timestamp'], 
        #                      infer_datetime_format=True, 
        #                      memory_map=True, 
        #                      index_col='Timestamp', 
        #                      low_memory=False)
            
        self.pass_history = 20
        self.strategy_obj = SampleStrategy()
        
        self.entry_price = None
        self.exit_price = None
        self.position = 0
        self.pnl = 0
        
        self.MA_period_far = MA_period_far
        self.MA_period_near = MA_period_near
        
        self.trade_id = -1
        self.trade_type = None
        self.entry_time = None
        self.exit_time = None
        self.exit_type = None
        
        self.data['MA NEAR'] = self.data['Close'].rolling(self.MA_period_near).mean()
        self.data['MA FAR'] = self.data['Close'].rolling(self.MA_period_far).mean()

        self.tradeLog = pd.DataFrame(columns=['Trade ID',
                                              'Trade Type',
                                              'Entry Time',
                                              'Entry Price',
                                              'Exit Time',
                                              'Exit Price',
                                              'PNL',
                                               ])
            
    def tradeExit(self):
        
        self.tradeLog.loc[self.trade_id, 'Trade ID'] = self.trade_id
        
        self.tradeLog.loc[self.trade_id, 'Trade Type'] = self.trade_type
                  
        self.tradeLog.loc[self.trade_id, 'Entry Time'] = pd.to_datetime(self.entry_time, infer_datetime_format= True)
        
        self.tradeLog.loc[self.trade_id, 'Entry Price'] = self.entry_price
                    
        self.tradeLog.loc[self.trade_id, 'Exit Time'] = pd.to_datetime(self.exit_time, infer_datetime_format= True)
        
        self.tradeLog.loc[self.trade_id, 'Exit Price'] = self.exit_price
        
        self.tradeLog.loc[self.trade_id, 'PNL'] = self.pnl*1000      
 
    def testerAlgo(self):

        def takeEntry():

            assert self.pass_history%1==0
            enterShortSignal =  self.strategy_obj.shortEntry(self.data.iloc[i-self.pass_history:i+1])
                
            enterLongSignal = self.strategy_obj.longEntry(self.data.iloc[i-self.pass_history:i+1])
            if enterShortSignal == True:
                self.position = -1
                self.trade_id = self.trade_id + 1
                self.trade_type = -1
                self.entry_time = self.data.index[i]
                self.entry_price = self.data['Close'][i]
                    
            elif enterLongSignal == True:
                self.position = 1 
                self.trade_id = self.trade_id + 1
                self.trade_type = 1
                self.entry_time = self.data.index[i]
                self.entry_price = self.data['Close'][i]
        
        for i in tqdm(range(self.pass_history, len(self.data)-1)):
            
            if self.position in [1, -1]:
                
                if self.position == -1:
                    assert self.pass_history%1==0
                    exitShortSignal =  self.strategy_obj.shortExit(self.data.iloc[i-self.pass_history:i+1])
                    
                    if exitShortSignal == True:
                        self.position = 0
                        self.exit_price = self.data['Close'][i]
                        self.pnl = (self.entry_price - self.exit_price)
                        self.exit_time = self.data.index[i]
                        self.tradeExit()
                        takeEntry()
                        
                if self.position == 1:

                    exitLongSignal =  self.strategy_obj.longExit(self.data.iloc[i-self.pass_history:i+1])
                        
                    if exitLongSignal == True:
                        self.position = 0
                        self.exit_price = self.data['Close'][i]
                        self.pnl = (self.exit_price - self.entry_price)
                        self.exit_time = self.data.index[i]
                        self.tradeExit()
                        takeEntry()

            elif self.position == 0:
                takeEntry()
                
                
                
class TestBroker():
    def __init__(self,
                 MA_period_far=200,
                 MA_period_near=50):
        
        url='https://drive.google.com/file/d/15m4eQ1OYO8tNQ8grAS57FIjRFhNHcOGG/view?usp=sharing'
        url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
        self.data = pd.read_csv(url2 ,
                             parse_dates=['Timestamp'], 
                             infer_datetime_format=True, 
                             memory_map=True, 
                             index_col='Timestamp', 
                             low_memory=False)
            
        self.pass_history = 20
        self.strategy_obj = SampleStrategy()
        
        self.entry_price = None
        self.exit_price = None
        self.position = 0
        self.pnl = 0
        
        self.MA_period_far = MA_period_far
        self.MA_period_near = MA_period_near
        
        self.trade_id = -1
        self.trade_type = None
        self.entry_time = None
        self.exit_time = None
        self.exit_type = None
        
        self.data['MA NEAR'] = self.data['Close'].rolling(self.MA_period_near).mean()
        self.data['MA FAR'] = self.data['Close'].rolling(self.MA_period_far).mean()

        self.tradeLog = pd.DataFrame(columns=['Trade ID',
                                              'Trade Type',
                                              'Entry Time',
                                              'Entry Price',
                                              'Exit Time',
                                              'Exit Price',
                                              'PNL',
                                               ])
            
    def tradeExit(self):
        
        self.tradeLog.loc[self.trade_id, 'Trade ID'] = self.trade_id
        
        self.tradeLog.loc[self.trade_id, 'Trade Type'] = self.trade_type
                  
        self.tradeLog.loc[self.trade_id, 'Entry Time'] = pd.to_datetime(self.entry_time, infer_datetime_format= True)
        
        self.tradeLog.loc[self.trade_id, 'Entry Price'] = self.entry_price
                    
        self.tradeLog.loc[self.trade_id, 'Exit Time'] = pd.to_datetime(self.exit_time, infer_datetime_format= True)
        
        self.tradeLog.loc[self.trade_id, 'Exit Price'] = self.exit_price
        
        self.tradeLog.loc[self.trade_id, 'PNL'] = self.pnl*1000      
 
    def testerAlgo(self):

        def takeEntry():

            assert self.pass_history%1==0
            enterShortSignal =  self.strategy_obj.shortEntry(self.data.iloc[i-self.pass_history:i+1])
                
            enterLongSignal = self.strategy_obj.longEntry(self.data.iloc[i-self.pass_history:i+1])
            if enterShortSignal == True:
                self.position = -1
                self.trade_id = self.trade_id + 1
                self.trade_type = -1
                self.entry_time = self.data.index[i]
                self.entry_price = self.data['Close'][i]
                    
            elif enterLongSignal == True:
                self.position = 1 
                self.trade_id = self.trade_id + 1
                self.trade_type = 1
                self.entry_time = self.data.index[i]
                self.entry_price = self.data['Close'][i]
        
        for i in tqdm(range(self.pass_history, len(self.data)-1)):
            
            if self.position in [1, -1]:
                
                if self.position == -1:
                    assert self.pass_history%1==0
                    exitShortSignal =  self.strategy_obj.shortExit(self.data.iloc[i-self.pass_history:i+1])
                    
                    if exitShortSignal == True:
                        self.position = 0
                        self.exit_price = self.data['Close'][i]
                        self.pnl = (self.entry_price - self.exit_price)
                        self.exit_time = self.data.index[i]
                        self.tradeExit()
                        takeEntry()
                        
                if self.position == 1:

                    exitLongSignal =  self.strategy_obj.longExit(self.data.iloc[i-self.pass_history:i+1])
                        
                    if exitLongSignal == True:
                        self.position = 0
                        self.exit_price = self.data['Close'][i]
                        self.pnl = (self.exit_price - self.entry_price)
                        self.exit_time = self.data.index[i]
                        self.tradeExit()
                        takeEntry()

            elif self.position == 0:
                takeEntry()

class Metrics():
    def __init__(self, 
                 trade_logs):
        
        self.trade_logs = trade_logs
        self.trade_logs['Entry Time'] = pd.to_datetime(self.trade_logs['Entry Time'], infer_datetime_format= True)
        self.trade_logs['Exit Time'] = pd.to_datetime(self.trade_logs['Exit Time'], infer_datetime_format= True)
        
        self.performance_metrics = pd.DataFrame(index=[
        'Total Trades',
        'Winning Trades',
        'Losing Trades',
        'Net P/L',
        'Gross Profit',
        'Gross Loss',
        'P/L Per Trade',
        'Max Drawdown',
        'Win Percentage',
        'Profit Factor'])
        
        self.monthly_performance = pd.DataFrame()
        
        self.yearly_performance = pd.DataFrame()
        
    def overall_calc(self):
        
            def total_trades_calc(self):
                return len(self.trade_logs)
    
            self.performance_metrics.loc['Total Trades', 'Overall'] = total_trades_calc(self)
            ################################################
            def winning_trades_calc(self):
                mask  = self.trade_logs['PNL']>0
                return len(self.trade_logs.loc[mask])
        
            self.performance_metrics.loc['Winning Trades', 'Overall'] = winning_trades_calc(self)
            ################################################
            def losing_trades_calc(self):
                mask  = self.trade_logs['PNL']<0
                return len(self.trade_logs.loc[mask])
        
            self.performance_metrics.loc['Losing Trades', 'Overall'] = losing_trades_calc(self)
            ################################################
            def gross_profit_calc(self):
                mask  = self.trade_logs['PNL']>0
                if len(self.trade_logs.loc[mask])>0:
                    return round(sum(self.trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Profit', 'Overall'] = gross_profit_calc(self)
            ################################################
            def gross_loss_calc(self):
                mask  = self.trade_logs['PNL']<0
                if len(self.trade_logs.loc[mask])>0:
                    return round(sum(self.trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Loss', 'Overall'] = gross_loss_calc(self)
            ################################################
            def net_pnl_calc(self):
                return round(sum(self.trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Net P/L', 'Overall'] = net_pnl_calc(self)
            ###############################################
            def pnl_per_trade_calc(self):
                return round(sum(self.trade_logs['PNL'])/len(self.trade_logs), 3)
        
            self.performance_metrics.loc['P/L Per Trade', 'Overall'] = pnl_per_trade_calc(self)
            ################################################
            def win_percentage_calc(self):
                return round((self.performance_metrics.loc['Winning Trades', 'Overall']/self.performance_metrics.loc['Total Trades', ('Overall')])*100,2)
        
            self.performance_metrics.loc['Win Percentage', 'Overall'] = win_percentage_calc(self)
            ################################################
            def profit_factor_calc(self):
                return round(abs(self.performance_metrics.loc['Gross Profit', 'Overall']/self.performance_metrics.loc['Gross Loss', ('Overall')]), 2)
        
            self.performance_metrics.loc['Profit Factor', 'Overall'] = profit_factor_calc(self)
            ################################################
            def pnl_per_win_calc(self):
                return round((self.performance_metrics.loc['Gross Profit', 'Overall']/self.performance_metrics.loc['Winning Trades', ('Overall')]),2)
        
            self.performance_metrics.loc['Profit Per Winning Trade', 'Overall'] = pnl_per_win_calc(self)
            ################################################
            def pnl_per_loss_calc(self):
                return round((self.performance_metrics.loc['Gross Loss', 'Overall']/self.performance_metrics.loc['Losing Trades', ('Overall')]),2)
        
            self.performance_metrics.loc['Loss Per Losing Trade', 'Overall'] = pnl_per_loss_calc(self)
            ################################################
            def max_drawdown_calc(self):
                xs = self.trade_logs['PNL'].cumsum()
                i = np.argmax(np.maximum.accumulate(xs) - xs)
                j = np.argmax(xs[:i])
                return round(abs(xs[i]-xs[j]),2)
        
            self.performance_metrics.loc['Max Drawdown', 'Overall'] = max_drawdown_calc(self)
            ################################################

            def monthly_perf_calc(self):
                return self.trade_logs.groupby(self.trade_logs['Entry Time'].dt.month)['PNL'].sum()
            
            self.monthly_performance['Overall'] = monthly_perf_calc(self)
            ###############################################
            def yearly_perf_calc(self):
                return self.trade_logs.groupby(self.trade_logs['Entry Time'].dt.year)['PNL'].sum()
            
            self.yearly_performance['Overall'] = yearly_perf_calc(self)
            ###############################################
        
    def plot_monthly_performance(self):
            fig = px.bar( y=self.monthly_performance['Overall'], x=self.monthly_performance.index, title='Monthly Performance')
            fig.show()
        
    def plot_yearly_performance(self, ):
            fig = px.bar( y=self.yearly_performance['Overall'], x=self.yearly_performance.index, title='Yearly Performance')
            fig.show()
        
    def plot_cumulative_returns(self):
            fig = px.line( y= self.trade_logs['PNL'].cumsum(), x=self.trade_logs['Entry Time'], title='Cumulative Returns')
            fig.show()
            
            
            
class GenerateSubmission():
    def __init__(self, 
                 train_trade_logs,
                 MA_period_far=None,
                 MA_period_near=None):
        
        assert MA_period_near is not None
        assert MA_period_far is not None
        
        self.MA_period_far = MA_period_far
        self.MA_period_near = MA_period_near
        
        self.train_trade_logs = train_trade_logs
        self.train_trade_logs['Entry Time'] = pd.to_datetime(self.train_trade_logs['Entry Time'], infer_datetime_format= True)
        self.train_trade_logs['Exit Time'] = pd.to_datetime(self.train_trade_logs['Exit Time'], infer_datetime_format= True)
        
        self.test_bt = TestBroker(MA_period_far=self.MA_period_far, MA_period_near=self.MA_period_near)
        
        self.test_bt.testerAlgo()
        
        self.test_trade_logs = self.test_bt.tradeLog
        
        self.combined_tradelog = pd.concat([self.test_trade_logs, self.train_trade_logs], axis=0)
        
        self.year_array = np.unique(self.combined_tradelog['Entry Time'].dt.year)
        
        self.submission_metrics = pd.DataFrame()
        
    def createSubmissionFile(self):
        
        for year in self.year_array:
            
            temp_tradelog = self.combined_tradelog.loc[self.combined_tradelog['Entry Time'].dt.year==year]
        
            def total_trades_calc(self):
                return len(temp_tradelog )
    
            self.total_trades = total_trades_calc(self)
            ################################################
            def winning_trades_calc(self):
                mask  = temp_tradelog['PNL']>0
                return len(temp_tradelog.loc[mask])
        
            self.winning_trades = winning_trades_calc(self)
            ################################################
            def net_pnl_calc(self):
                return round(sum(temp_tradelog['PNL']),2)
        
            self.submission_metrics.loc[str(year), 'Net P/L'] = net_pnl_calc(self)
            ###############################################
            def pnl_per_trade_calc(self):
                return round(sum(temp_tradelog['PNL'])/len(temp_tradelog), 3)
        
            self.submission_metrics.loc[str(year), 'Avg P/L'] = pnl_per_trade_calc(self)
            ################################################
            def win_percentage_calc(self):
                return round((self.winning_trades/self.total_trades)*100,2)
        
            self.submission_metrics.loc[str(year),  'Win %'] = win_percentage_calc(self)
            ################################################
            
        self.submission_metrics.index.name = 'Year'
