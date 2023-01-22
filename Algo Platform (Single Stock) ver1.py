import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import CheckButtons
from mycolorpy import colorlist as mcp
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import pandas_ta as ta
import datetime
import math
import numpy as np
#from dt_auto import read_csv

def figure_design(axs):
    for ax in axs:
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(axis='both', labelsize=14, colors='#e4e4e4')
        ax.ticklabel_format(useOffset=False)
        ax.spines['bottom'].set_color('#787878')
        ax.spines['top'].set_color('#787878')
        ax.spines['left'].set_color('#787878')
        ax.spines['right'].set_color('#787878')

def ax_design(ax, y_axis_visible = False, x_axis_label = False):
    ax.clear()
    ax.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)

    if y_axis_visible == False:
        ax.axes.yaxis.set_visible(y_axis_visible)
    else:
        ax.yaxis.set_ticks_position("right")

    if x_axis_label == False:
        ax.set_xticklabels([])
    else:
        ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticklabels([])

def compute_plot_TA(ax, data, showMA=False, MAs=[], showEMA=False, EMAs=[], showBB = False, BB = []):
    if (len(MAs) + len(EMAs)) > 0:
        colorlist = mcp.gen_color(cmap='tab20', n = len(MAs) + len(EMAs))

    if (showMA == True) & (len(MAs)>0):
        for MA in MAs:
            name = 'MA' + str(MA)
            data[name] = data["close"].rolling(MA).mean()
            ax.plot(data[name], color=colorlist[0], linestyle="-", linewidth=1, label= str(MA) + ' periods SMA')
            colorlist.pop(0)

    if (showEMA == True) & (len(EMAs)>0):
        for EMA in EMAs:
            name = 'EMA' + str(EMA)
            data[name] = data["close"].ewm(span=EMA, adjust=False).mean()
            ax.plot(data[name], color=colorlist[0], linestyle="-", linewidth=1, label= str(EMA) + ' periods EMA')
            colorlist.pop(0)

    if (showBB == True) & (len(BB) == 2):
        bb = ta.bbands(data['close'], length=BB[0], std=BB[1])
        if bb is not None:
            ax.fill_between(bb.index, bb.iloc[:, 2], bb.iloc[:, 0],
                         facecolor='#666699', alpha=0.2,label='Bollinger Bands')
            ax.plot(bb.iloc[:, 2], color='#666699', linestyle="-", linewidth=0.2)
            ax.plot(bb.iloc[:, 0], color='#666699', linestyle="-", linewidth=0.2)

    return data

def compute_plot_OHLC(ax, data):
    candle_counter = range(len(data['open']))
    ohlc = []
    for i in candle_counter:
        append_me = candle_counter[i], data["open"][i], data["high"][i], data["low"][i], data["close"][i]
        ohlc.append(append_me)

    candlestick_ohlc(ax, ohlc, width=0.8, colorup='#53b987', colordown='#eb4d5c')

    if data['close'].iloc[-1]>=data['open'].iloc[-1]:
        colorcode = '#53b987'
    else:
        colorcode = '#eb4d5c'

    #########################################################################################################
    ax.axhline(data['close'].iloc[-1], linestyle='--', color=colorcode, linewidth=0.5)

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.005, data['close'].iloc[-1], data['close'].iloc[-1], color="#e4e4e4", fontsize=12,
            transform=trans, horizontalalignment='left', verticalalignment='center',
            bbox=dict(facecolor=colorcode, edgecolor=colorcode))
    #########################################################################################################

    strings = ['O', str(data['open'].iloc[-1]),
               'H', str(data['high'].iloc[-1]),
               'L', str(data['low'].iloc[-1]),
               'C', str(data['close'].iloc[-1])]

    colors = ['#e4e4e4', colorcode,
              '#e4e4e4', colorcode,
              '#e4e4e4', colorcode,
              '#e4e4e4', colorcode]
    margin_label = 0
    margin_price = 0
    for s, c in zip(strings, colors):
        ax.text(0.75 + margin_label + margin_price, 0.95, s + " ", color=c, transform=ax.transAxes, fontsize=12,
                fontweight='bold',
                horizontalalignment='left', verticalalignment='center')
        if c == '#e4e4e4':
            margin_label = margin_label + 0.01
        else:
            margin_price = margin_price + 0.05

    return ohlc

def plot_header(ax, stock_code, latest_price, latest_change, target):
    ax.text(0.12, 0.95, stock_code, transform=ax.transAxes, color='#e4e4e4',
             fontsize=12, fontweight='bold',
             horizontalalignment='left', verticalalignment='center',)

    ax.text(0.18, 0.95, target, transform=ax.transAxes, color='#08a0e9', fontsize=12, fontweight='bold',
             horizontalalignment='left', verticalalignment='center')

    ax.text(0.12, 0.90, latest_price, transform=ax.transAxes, color='#e4e4e4', fontsize=12, fontweight='bold',
             horizontalalignment='left', verticalalignment='center')

    if latest_change[0]=='+':
        colorcode = '#53b987'
    else:
        colorcode = '#eb4d5c'
    ax.text(0.17, 0.90, latest_change, transform=ax.transAxes, color=colorcode, fontsize=12, fontweight='bold',
             horizontalalignment='left', verticalalignment='center')

    time_stamp = datetime.datetime.now()
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.93, 1.05, time_stamp, transform=ax.transAxes, color='white', fontsize=12, fontweight='bold',
             horizontalalignment='center', verticalalignment='center')

    ax.text(-0.07, 0.94, 'Indicators', transform=ax.transAxes, color='white', fontsize=10, fontweight='bold',
            horizontalalignment='left', verticalalignment='center')

    ax.text(-0.07, 0.66, 'Strategy', transform=ax.transAxes, color='white', fontsize=10, fontweight='bold',
            horizontalalignment='left', verticalalignment='center')

def plot_volume(ax, data, vol):
    pos = data['open'] - data['close'] <= 0
    neg = data['open'] - data['close'] > 0

    ax.bar(data.index[pos], data['volume_diff'][pos], color='#53b987', width=0.8, align='center')
    ax.bar(data.index[neg], data['volume_diff'][neg], color='#eb4d5c', width=0.8, align='center')

    ymax = data['volume_diff'].max()
    ystd = data['volume_diff'].std()
    if not (math.isnan(ymax) and math.isnan(ystd)) and ymax!=0:
        ax.set_ylim([0, ymax + ystd])

    ax.text(0.01, 0.95, 'Volume: ' + "{:,}".format(int(vol)), transform=ax.transAxes, color='#e4e4e4',
            fontsize=10, fontweight='bold',
            horizontalalignment='left', verticalalignment='top')

def plot_MACD(ax, data):
    if len(data["close"])>33:
        macd = ta.macd(data["close"]).fillna(0)
        data = pd.concat([data, macd], axis=1).reindex(data.index)

        # 2nd change
        ax.plot(data.index, np.where(data['MACD_12_26_9'] == 0, data['MACD_12_26_9'], None), label='MACD', linewidth=1, alpha = 0)
        ax.plot(data.index, np.where(data['MACD_12_26_9'] != 0, data['MACD_12_26_9'], None), label='MACD', linewidth=1, color='white')

        ax.plot(data.index, np.where(data['MACDs_12_26_9'] == 0, data['MACDs_12_26_9'], None), label='signal', linewidth=1,  alpha = 0)
        ax.plot(data.index, np.where(data['MACDs_12_26_9'] != 0, data['MACDs_12_26_9'], None) , label='signal', linewidth=1, color='orange')

        pos = data['MACDh_12_26_9'] >= 0
        neg = data['MACDh_12_26_9'] < 0

        ax.bar(data.index[pos], data['MACDh_12_26_9'][pos], color='#53b987', width=0.8, align='center')
        ax.bar(data.index[neg], data['MACDh_12_26_9'][neg], color='#eb4d5c', width=0.8, align='center')

        if len(data['MACD_12_26_9']) != 0:
            ax.text(0.01, 0.95, 'MACD(12, 26, 9)', transform=ax.transAxes, color='white',
                    fontsize=10, fontweight='bold',
                    horizontalalignment='left', verticalalignment='top')

def plot_RSI(ax, data):
    ax.set_ylim([0,100])
    ax.axhline(30, linestyle='-', color='green', linewidth=0.5)
    ax.axhline(50, linestyle='-', alpha=0.5, color='white', linewidth=0.5)
    ax.axhline(70, linestyle='-', color='red', linewidth=0.5)
    ax.plot(data.index, np.where(data['RSI'] == 0, data['RSI'], None), color='#37a6ef', alpha = 0)
    ax.plot(data.index, np.where(data['RSI'] != 0, data['RSI'], None), color='#37a6ef', linewidth=1.5)
    ax.bar(data.index, data['RSI'], color='#53b987', width=0.8, align='center',  alpha = 0)
    if len(data['RSI'])!=0:
        ax.text(0.01, 0.95, 'RSI(14): ' + str(round(data['RSI'].iloc[-1],2)), transform=ax.transAxes,
                color='#e4e4e4', fontsize=10, fontweight='bold',
                horizontalalignment='left', verticalalignment='top')

def plot_x_axis_time(ax, data):
    xdate = [i for i in data['time']]
    def mydate(x, pos=None):
        try:
            t = xdate[int(x)].strftime('%H:%M')
            return xdate[int(x)].strftime('%H:%M')
        except IndexError:
            return ''
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(mydate))

def check_convert_str_float(df, column):
    if isinstance(df[column][0], str):
        df[column] = df[column].str.replace(',', '')
        df[column] = df[column].str.replace('[', '', regex=True)
        df[column] = df[column].replace(']', np.NaN, regex=True)
        df[column] = df[column].astype(float)
    return df

def process_data(filename, stock_code, usecols):
    df = pd.read_csv(filename, header=None,  usecols = usecols,#sep="\t", 
                     names=['time', stock_code, 'change', 'volume', 'target'],
                     index_col=['time'], parse_dates=['time'])


    index_with_nan = df.index[df.isnull().any(axis=1)]
    df.drop(index_with_nan, axis=0, inplace=True)

    df.index = pd.DatetimeIndex(df.index)

    #df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
    df = check_convert_str_float(df, stock_code)
    df = check_convert_str_float(df, 'volume')
    df = check_convert_str_float(df, 'target')

    df.fillna(method='ffill', inplace=True)

    latest_info = df.iloc[-1, :]
    latest_price = str(latest_info.iloc[0])
    latest_change = str(latest_info.iloc[1])

    ########################################################################################
    df_vol=df['volume'].resample('1Min').mean()

    # - Convert tick data to ohlc format
    data = df[stock_code].resample("1Min").ohlc()


        #--------------------Change Made

    data['time'] = data.index
    #data = data.set_index('time')
    data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d %H:%M:%S")
    

    # problems if data is less than 20
    data['RSI'] = ta.rsi(data['close'], timeperiod=14)
    data['RSI'] = data['RSI'].fillna(0)

    data['volume_diff'] = df_vol.diff().fillna(0)
    data.loc[data['volume_diff']<0, 'volume_diff']=0

    data.reset_index(drop=True, inplace=True)

    return data, latest_price, latest_change, df['target'].iloc[-1], df['volume'].iloc[-1]

def interactive_TA():
    check = plt.axes([0.07, 0.73, 0.05, 0.1])  # x, y, width, height
    figure_design([check])
    check.set_facecolor('#121416')

    # Define colours for rectangles and set them
    activated = [True, True, True]
    plot_button = CheckButtons(check, ['SMA', 'EMA', 'BB'], activated)
    for r in plot_button.rectangles:
        r.set_facecolor("w")
        r.set_edgecolor("w")
    [ll.set_color("#37a6ef") for l in plot_button.lines for ll in l]
    [ll.set_linewidth(1.5) for l in plot_button.lines for ll in l]
    for i, c in enumerate(["w", "w", "w"]):
        plot_button.labels[i].set_color(c)
        plot_button.labels[i].set_color("w")
        plot_button.labels[i].set_fontsize(12)
    return plot_button

def interactive_strategy():
    check = plt.axes([0.07, 0.58, 0.05, 0.1])  # x, y, width, height
    figure_design([check])
    check.set_facecolor('#121416')

    # Define colours for rectangles and set them
    activated = [False, False, False, False]
    plot_button = CheckButtons(check, ['SMA', 'MACD', 'RSI', 'BB'], activated)
    for r in plot_button.rectangles:
        r.set_facecolor("w")
        r.set_edgecolor("w")
    [ll.set_color("#37a6ef") for l in plot_button.lines for ll in l]
    [ll.set_linewidth(1.5) for l in plot_button.lines for ll in l]
    for i, c in enumerate(["w", "w", "w", "w"]):
        plot_button.labels[i].set_color(c)
        plot_button.labels[i].set_color("w")
        plot_button.labels[i].set_fontsize(12)
    return plot_button

def MA_Strategy(ax, data):
    data['MA5'] = data["close"].rolling(5).mean()
    data['MA10'] = data["close"].rolling(10).mean()
    data['MA20'] = data["close"].rolling(20).mean()

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['MA20'][i]):
            if (data['MA5'][i] > data['MA10'][i]) & (data['MA5'][i] > data['MA20'][i]):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif (data['MA5'][i] < data['MA10'][i]) & (data['MA5'][i] < data['MA20'][i]):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['high'][i] + data['high'][i] * 0.001)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['high'][i] + data['high'][i] * 0.001)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    data['MA Buy'] = Buy
    data['MA Sell'] = Sell

    ax.scatter(data.index, data['MA Buy'], label='MA Buy', marker=r'$\spadesuit$',
               facecolors='none', edgecolors='#00FFFF', alpha=1, s=150)
    ax.scatter(data.index, data['MA Sell'], label='MA Sell', marker=r'$\spadesuit$',
               facecolors='none', edgecolors='#FFFF00', alpha=1, s=150)

    return data, Record

def MACD_Strategy(ax, data):
    macd = ta.macd(data["close"])*100
    macd.rename(columns={'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'Histogram', 'MACDs_12_26_9': 'Signal'}, inplace=True)
    data = pd.concat([data, macd], axis=1).reindex(data.index)

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['Histogram'][i - 1]):
            if ((data['Histogram'][i - 1]<0) & (data['Histogram'][i]>0)) & \
                ((data['MACD'][i]<0) & (data['Signal'][i]<0)):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif ((data['Histogram'][i - 1]>0) & (data['Histogram'][i]<0)) & \
                ((data['MACD'][i]>0) & (data['Signal'][i]>0)):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['high'][i] + data['high'][i] * 0.001)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['high'][i] + data['high'][i] * 0.001)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    data['MACD Buy'] = Buy
    data['MACD Sell'] = Sell

    ax.scatter(data.index, data['MACD Buy'], label='MACD Buy', marker=r'$\heartsuit$',
               facecolors='none', edgecolors='#00FFFF', alpha=1, s=150)
    ax.scatter(data.index, data['MACD Sell'], label='MACD Sell', marker=r'$\heartsuit$',
               facecolors='none', edgecolors='#FFFF00', alpha=1, s=150)

    return data, Record

def RSI_Strategy(ax, data):
    data['RSI'] = ta.rsi(data['close'], timeperiod=14)

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['RSI'][i - 1]):
            if ((data['RSI'][i - 1]<30) & (data['RSI'][i]>30)):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif ((data['RSI'][i - 1]>70) & (data['RSI'][i]<70)):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['low'][i] + data['low'][i] * 0.001)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['low'][i] + data['low'][i] * 0.001)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    data['RSI Buy'] = Buy
    data['RSI Sell'] = Sell

    ax.scatter(data.index, data['RSI Buy'], label='RSI Buy', marker=r'$\clubsuit$',
               facecolors='none', edgecolors='#00FFFF', alpha=1, s=150)
    ax.scatter(data.index, data['RSI Sell'], label='RSI Sell', marker=r'$\clubsuit$',
               facecolors='none', edgecolors='#FFFF00', alpha=1, s=150)

    return data, Record

def BB_Strategy(ax, data):
    bb = ta.bbands(data['close'], length=20, std=2)
    bb.rename(columns={'BBU_20_2.0': 'BBU', 'BBL_20_2.0': 'BBL'}, inplace = True)
    data = pd.concat([data, bb], axis=1).reindex(data.index)

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['BBU'][i]):
            if (data['BBL'][i] > (data['close'][i] + data['open'][i])/2):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['low'][i] - data['low'][i] * 0.001)
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif (data['BBU'][i] < (data['close'][i]+data['open'][i])/2):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['high'][i] + data['high'][i] * 0.001)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['high'][i] + data['high'][i] * 0.001)
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    data['BB Buy'] = Buy
    data['BB Sell'] = Sell

    ax.scatter(data.index, data['BB Buy'], label='BB Buy', marker=r'$\diamondsuit$',
               facecolors='none', edgecolors='#00FFFF', alpha=1, s=150)
    ax.scatter(data.index, data['BB Sell'], label='BB Sell', marker=r'$\diamondsuit$',
               facecolors='none', edgecolors='#FFFF00', alpha=1, s=150)

    return data, Record

def animate(i):
    time_stamp = datetime.datetime.now() - datetime.timedelta(hours=12)
    time_stamp = time_stamp.strftime("%Y-%m-%d")
    filename = str(time_stamp) + ' stock data.csv'
    data, latest_price, latest_change, target, vol = process_data(filename, Stock[0], [1,2,3,4,5])

    ##############################################################################################################
    # Main ##########################################################################
    ax_design(ax1, y_axis_visible=True, x_axis_label=False)
    # plot header
    plot_header(ax1, Stock[0], latest_price, latest_change, target)

    # compute & plot TA
    showMA_status, showEMA_status, showBB_status = plot_button_TA.get_status()
    data = compute_plot_TA(ax1, data, showMA=showMA_status, MAs=[5, 10, 20], showEMA=showEMA_status, EMAs=[20], showBB=showBB_status, BB=[20, 2])

    MA_status, MACD_status, RSI_status, BB_status = plot_button_strategy.get_status()
    if (MA_status == True) & (len(data['MA20'])>20):
        if (data['MA20'].iloc[-1]!=data['MA10'].iloc[-1]) & (data['MA20'].iloc[-1]!=data['MA5'].iloc[-1]):
            MA_Strategy(ax1, data)
    if (MACD_status == True) & (len(data['close'])>26):
        MACD_Strategy(ax1, data)
    if (RSI_status == True) & (data['RSI'].iloc[-1]>0):
        RSI_Strategy(ax1, data)
    if (BB_status == True) & (len(data['close'])>20):
        BB_Strategy(ax1, data)

    # compute & plot ohlc
    ohlc = compute_plot_OHLC(ax1, data)

    if (showMA_status) | (showEMA_status) | (showBB_status):
        leg = ax1.legend(loc='upper left', facecolor='#121416', fontsize=10)
        plt.setp(leg.get_texts(), color='w')

    ##############################################################################################################
    # Sub-volume ##########################################################################
    ax_design(ax2, y_axis_visible=False, x_axis_label=False)
    plot_volume(ax2, data, vol)

    ##############################################################################################################
    # Sub-MACD ##########################################################################
    ax_design(ax3, y_axis_visible=True, x_axis_label=False)
    plot_MACD(ax3, data)
    ##############################################################################################################
    # Sub-RSI ##########################################################################
    ax_design(ax4, y_axis_visible=True, x_axis_label=True)
    ax4.axes.yaxis.set_ticks([30, 70])
    plot_RSI(ax4, data)
    plot_x_axis_time(ax4, data)
    ##############################################################################################################

########################################################################################
########################################################################################
fig = plt.figure()
fig.patch.set_facecolor('#121416')
gs = fig.add_gridspec(10, 6)
ax1 = fig.add_subplot(gs[0:7, 0:6])
ax2 = fig.add_subplot(gs[7, 0:6])
ax3 = fig.add_subplot(gs[8, 0:6])
ax4 = fig.add_subplot(gs[9, 0:6])
figure_design([ax1, ax2, ax3, ax4])

Stock=['TataPower']
plot_button_TA = interactive_TA()
plot_button_strategy = interactive_strategy()

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()