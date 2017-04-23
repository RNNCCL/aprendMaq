# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import urllib2 as url
import datetime as dt
import json
import sys
#import matplotlib.pyplot as plt
from sklearn import svm
'''
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
    '''
summary={'summary':[]}
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
    return np.asarray(X),np.asarray(y)
    

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

def split_data_set_multiple_training_split(time_serie,training,validation):
    time_serie_len = len(time_serie.values)
    delta_training = training*time_serie_len - int(training*time_serie_len)
    delta_validation = validation*time_serie_len - int(validation*time_serie_len)
    validation_size=0
    if delta_training > delta_validation:
        training_size = int(training*time_serie_len)+1
        validation_size = int(validation*time_serie_len)+1
    else:
        training_size = int(training*time_serie_len)
        validation_size = int(validation*time_serie_len)
    return time_serie[0:training_size+1], time_serie[training_size+1:training_size+1+validation_size]
    

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

def predict_and_validate(trained_model, test_array,window_size,n=0):
    results_list = []
    if len(test_array) == window_size:
    	result = result = trained_model.predict(test_array)[0]
    	results_list = results

    elif n != 0:
    	print "IF"
    	for index,i in enumerate(test_array):
    		if (index+window_size+n) > len(test_array)-1:
    			break
    		result = trained_model.predict(test_array[index:index+window_size])[0]
    		expected = test_array[n]
    		delta_abs = abs(test_array[n]-expected)
    		delta = test_array[n]-expected
    		results_list.append({'expected':expected,'predicted':result,'euclidian_distance':(((expected-result) ** 2) ** (0.5)),'error_rate':(abs(expected-result)/(expected+result))})
    else:
    	print "ELSE"
    	for index,i in enumerate(test_array):
    		if (index+window_size+1) > len(test_array)-1:
    			break
    		result = trained_model.predict(test_array[index:index+window_size])[0]
    		expected = test_array[index+window_size+1]
    		delta_abs = abs(test_array[n]-expected)
    		delta = test_array[n]-expected
    		results_list.append({'expected':expected,'predicted':result,'euclidian_distance':(((expected-result) ** 2) ** (0.5)),'error_rate':(abs(expected-result)/(expected+result))})
    return {'validation_test':results_list}

def train_test_split(training_size,validation_size,window_size,c,kernel,k_fold=None):
    if k_fold:
        result={'result':[]}
        time_serie = import_data()
        training_set, validation_set = split_data_set(time_serie,training_size,validation_size)    
        X,y=set_bases(training_set,window_size)
        C = c
        validation_array = []
        for i in validation_set.values:
            validation_array.append(i[0])
        print "INIT TRAIN TEST SPLIT", dt.datetime.now()
        svr = svm.SVR(kernel=kernel, C=C).fit(X, y)
        validated_array = predict_and_validate(svr,validation_array,window_size)
        df = pd.DataFrame(validated_array['validation_test'])
        df.to_csv('k_fold_'+str(training_size)+'_'+str(validation_size)+'_'+str(window_size)+'_'+str(kernel)+'_'+str(c)+'.csv')
        summary['summary'].append({'validation_type':'unit k-fold','kernel':kernel,'C':c,'euclidian_distance':df['euclidian_distance'].mean(),'window_size':window_size,'error_rate':df['error_rate'].mean()})
        #result['result'].append({'validation_type':'train test split','kernel':kernel,'C':c,'euclidian_distance':df['euclidian_distance'].mean(),'window_size':window_size})
        print "END TRAIN TEST SPLIT", dt.datetime.now()
        return [df['euclidian_distance'].mean(),df['error_rate'].mean()]
    else:
        time_serie = import_data()
        training_set, validation_set = split_data_set(time_serie,training_size,validation_size)    
        X,y=set_bases(training_set,window_size)
        C = c
        validation_array = []
        for i in validation_set.values:
        	validation_array.append(i[0])
        print "INIT TRAIN TEST SPLIT", dt.datetime.now()
        svr = svm.SVR(kernel=kernel, C=C).fit(X, y)
        validated_array = predict_and_validate(svr,validation_array,window_size)
        df = pd.DataFrame(validated_array['validation_test'])
        df.to_csv('train_test_split_'+str(training_size)+'_'+str(validation_size)+'_'+str(window_size)+'_'+str(kernel)+'_'+str(c)+'.csv')
        summary['summary'].append({'validation_type':'train test split','kernel':kernel,'C':c,'euclidian_distance':df['euclidian_distance'].mean(),'window_size':window_size,'error_rate':df['error_rate'].mean()})
        print "END TRAIN TEST SPLIT", dt.datetime.now()
        return

def multiple_train_test_split(training_array,window_size,c,kernel):
    time_serie = import_data()
    result=[]
    for index,i in enumerate(training_array):
    	training_size = i[0]
    	validation_size=i[1]
    	training_set, validation_set = split_data_set_multiple_training_split(time_serie,training_size,validation_size)    
        
    	X,y=set_bases(training_set,window_size)
    	C = c
    	validation_array = []
    	for i in validation_set.values:   
    		validation_array.append(i[0])
    	print "INIT TRAIN TEST SPLIT", dt.datetime.now()
    	svr = svm.SVR(kernel=kernel, C=C).fit(X, y)
    	validated_array = predict_and_validate(svr,validation_array,window_size)
    	df = pd.DataFrame(validated_array['validation_test'])
        df.to_csv('multiple_train_test_split_'+str(training_size)+'_'+str(validation_size)+'_'+str(window_size)+'_'+str(kernel)+'_'+str(c)+'.csv')
    	#summary['summary'].append({'validation_type':'unit multiple train test split','kernel':kernel,'C':c,'euclidian_distance':df['euclidian_distance'].mean(),'window_size':window_size,'error_rate':df['error_rate'].mean()})
        result.append([df['euclidian_distance'].mean(),df['error_rate'].mean()])

    	print "END TRAIN TEST SPLIT", dt.datetime.now()
    euclidian_distance=0
    error_mean=0
    for i in result:
        euclidian_distance+=i[0]
        error_mean+=i[1]
    euclidian_distance=euclidian_distance/len(result)
    error_mean = error_mean/len(result)
    summary['summary'].append({'validation_type':'mean multiple train test split','kernel':kernel,'C':c,'euclidian_distance':euclidian_distance,'window_size':window_size,'error_rate':error_mean})
    return

def walking_forward(window_size,c,kernel):
    time_serie = import_data()
    result_array=[]
    dict_result={'result':[]}
    for index,i in enumerate(time_serie.values):
        if index+window_size+1+window_size+1 > len(time_serie.values)-1:
            break
        X=time_serie.values[index:index+window_size]
        y=time_serie.values[index+window_size+1]
        predict=(time_serie.values[index+window_size+1:index+window_size+window_size][0])
        expected = time_serie.values[index+window_size+window_size+1][0]
        C = c
        print "INIT TRAIN TEST SPLIT", dt.datetime.now()
        svr = svm.SVR(kernel=kernel, C=C).fit((X[0],1),y )
        print 'preditc', predict
        prediction = svr.predict((predict,1))
        result_array.append([(((expected-prediction) ** 2) ** (0.5)),(abs(expected-prediction)/(expected+prediction))])
        dict_result['result'].append({'validation_type':'walking forward','kernel':kernel,'C':c,'euclidian_distance':(((expected-prediction[0]) ** 2) ** (0.5)),'error_rate':(abs(expected-prediction[0])/(expected+prediction[0]))})

    df = pd.DataFrame(dict_result['result'])
    df.to_csv('walking_forward_'+str(window_size)+'_'+str(kernel)+'_'+str(c)+'.csv')
    mean_euclidian_distance = 0
    mean_error_rate=0
    for i in result_array:
        mean_euclidian_distance+=i[0][0]
        mean_error_rate+=i[1][0]
    mean_euclidian_distance = mean_euclidian_distance/len(result_array)
    mean_error_rate = mean_error_rate/len(result_array)
    summary['summary'].append({'validation_type':'mean walking forward','kernel':kernel,'C':c,'euclidian_distance':mean_euclidian_distance,'window_size':window_size,'error_rate':mean_error_rate})
    print "END TRAIN TEST SPLIT", dt.datetime.now()
    return

def rolling_window(window_size,c,kernel):
    start_windows_size = window_size
    time_serie = import_data()
    result_array=[]
    dict_result={'result':[]}
    for index,i in enumerate(time_serie.values):
        window_size+=1
        X=[]
        y=[]
        if index+window_size+1+window_size+1 > len(time_serie.values)-1:
            break
        X=time_serie.values[index:index+window_size]
        y=time_serie.values[index+window_size+1]
        predict=(time_serie.values[index+window_size+1:index+window_size+window_size][0])
        expected = time_serie.values[index+window_size+window_size+1][0]
        temp=[]
        C = c
        print "INIT TRAIN TEST SPLIT", dt.datetime.now()
        svr = svm.SVR(kernel=kernel, C=C).fit((X[0],1),y )
        prediction = svr.predict((predict,1))
        result_array.append([(((expected-prediction) ** 2) ** (0.5)),(abs(expected-prediction)/(expected+prediction))])
        dict_result['result'].append({'validation_type':'rolling window','kernel':kernel,'C':c,'euclidian_distance':(((expected-prediction[0]) ** 2) ** (0.5)),'error_rate':(abs(expected-prediction[0])/(expected+prediction[0]))})

    df = pd.DataFrame(dict_result['result'])
    df.to_csv('rolling window'+'_'+str(window_size)+'_'+str(kernel)+'_'+str(c)+'.csv')
    mean_euclidian_distance = 0
    mean_error_rate=0
    for i in result_array:
        mean_euclidian_distance+=i[0][0]
        mean_error_rate+=i[1][0]
    mean_euclidian_distance = mean_euclidian_distance/len(result_array)
    mean_error_rate = mean_error_rate/len(result_array)
    summary['summary'].append({'validation_type':'mean rolling window','kernel':kernel,'C':c,'euclidian_distance':mean_euclidian_distance,'window_size':start_windows_size,'error_rate':mean_error_rate})
    print "END TRAIN TEST SPLIT", dt.datetime.now()
    return

def k_fold(training_array,window_size,c,kernel):
    result_array=[]
    for i in training_array:
        training_size=i[0]
        validation_size=i[1]
        result_array.append(train_test_split(training_size,validation_size,window_size,c,kernel,k_fold=True))
    euclidian_median=0
    error_rate = 0
    for i in result_array:
        euclidian_median+=i[0]
        error_rate+=i[1]
    euclidian_median = euclidian_median/len(result_array)
    error_rate = error_rate/len(result_array)
    summary['summary'].append({'validation_type':'mean k-fold','kernel':kernel,'C':c,'euclidian_distance':euclidian_median,'window_size':window_size,'error_rate':error_rate})
    return

def main():
    kernel_list=['linear','polynomial','sigmoid']
    window_size_one=[5,10,20,50]
    window_size_two=[50,100,200,500]
    for i in kernel_list:
        for j in window_size_one:
            print "train_test_split"
            train_test_split(0.8,0.2,j,1.0,i)
            print "multiple_train_test_split"
            multiple_train_test_split([[0.6,0.2],[0.7,0.2],[0.8,0.2]],j,1.0,i)
            print "k_fold"
            k_fold([[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3]],j,1.0,i)
            k_fold([[0.3,0.7],[0.4,0.6]],j,1.0,i)
        for j in window_size_two:
            print "walking_forward"
            walking_forward(j, 1.0,i)
            print 'rolling_window'
            rolling_window(j, 1.0,i)
    df_summary = pd.DataFrame(summary['summary'])
    df_summary.to_csv('summary.csv')

if __name__=='__main__':
    main()