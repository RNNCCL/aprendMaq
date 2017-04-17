# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import urllib2 as url
import datetime as dt
import json
import sys
import matplotlib.pyplot as plt
from sklearn import svm

def set_bases(training_set,window_size):
    X = []
    y = []
    temp = []
    for index,i in enumerate(training_set.values):
        temp=[]
        if(index+window_size+1) == len(training_set.values):
            for j in range(0,window_size+1):
                if j == window_size:
                    y.append(training_set.values[index+j][0])
                else:
                    temp.append(training_set.values[index+j][0])
            X.append(temp)
            break
        else:   
            for j in range(0,window_size+1):
                if j == window_size:
                    y.append(training_set.values[index+j][0])
                else:
                    temp.append(training_set.values[index+j][0])
            X.append(temp)   
    return X,y
    

def format_date(dt):
    year = str(dt.year)
    if len(str(dt.month)) == 1:
        month  = '0'+str(dt.month)
    else:
        month = str(dt.month)
    if len(str(dt.day)) == 1:
        day  = '0'+str(dt.day)
    else:
        day = str(dt.day)
    return year+'-'+month+'-'+day


def split_data_set(time_serie,training,validation):
    time_serie_len = len(time_serie.values)
    delta_training = training*time_serie_len - int(training*time_serie_len)
    delta_validation = validation*time_serie_len - int(validation*time_serie_len)
    if delta_training > delta_validation:
        training_size = int(training*time_serie_len)+1
    else:
        training_size = int(training*time_serie_len)
    return time_serie[0:training_size+1], time_serie[training_size+1:]
    

def import_data():
    today = dt.datetime.now()
    print "API Call address"
    print 'http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-18&end='+format_date(today)
    json_data = url.urlopen('http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-18&end='+format_date(today)).read()
    df = pd.DataFrame(json.loads(json_data))
    del df['disclaimer']
    del df['time']
    df = df.drop(df.index[len(df.values)-2])
    df = df.drop(df.index[len(df.values)-1])
    df.to_csv('bitcoin.csv')
    return df
    
def main():
    time_serie = import_data()
    #time_serie, training set size, validation set size
    training_set, validation_set = split_data_set(time_serie,0.7,0.3)    
    X,y=set_bases(training_set,5)
    X = np.asarray(X)
    y = np.asarray(y) 
    C = 1.0
    teste = []
    for i in validation_set.values[0:5]:
        teste.append(i[0])
    print "INIT", dt.datetime.now()
    svr = svm.SVR(kernel='linear', C=C).fit(X, y)
    prediction = svr.predict(teste)    
    time_serie.plot()
    print time_serie
    plt.show()
    print "PREDICTED RESULT", prediction
    print "EXPECTED RESULT", validation_set.values[6]
    print "DIF", validation_set.values[6][0] - prediction
    print "RATE",prediction / validation_set.values[6]
    print "END", dt.datetime.now()
if __name__=='__main__':
    main()

