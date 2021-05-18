# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:22:20 2021

@author: Lycanthrope
"""
import pandas as pd
import numpy as np

#weather data 
filename_weather = 'weather_train_set4.csv'

df_weather = pd.read_csv(filename_weather)

df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

df_weather['year'], df_weather['month'], df_weather['day'],df_weather['hour']= df_weather['datetime'].dt.year, df_weather['datetime'].dt.month,df_weather['datetime'].dt.day, df_weather['datetime'].dt.hour

cols = range(1,13)
attributeNames = np.asarray(df_weather.columns[cols])

weather_data = np.asarray(df_weather[attributeNames])
Hour = np.asarray(df_weather['hour'])
Hour = Hour[:,np.newaxis]


#PV data
filename_solar = 'pv_train_set4.csv'

df_solar = pd.read_csv(filename_solar)

print(df_solar.isnull().any())

cols = [1,3,2]
attributeNames2 = np.asarray(df_solar.columns[cols])

solar_data = np.asarray(df_solar[attributeNames2])

solar_hour = np.zeros((len(Hour),3))
j=0
for i in range(0,len(solar_data),2):
    for k in range(3):
        if solar_data[i,k] !=np.nan and solar_data[i+1,k] !=np.nan:
            solar_hour[j,k]= (solar_data[i,k] + solar_data[i+1,k])/2
        else:
            solar_hour[j,k]= np.nan
    j+=1

#load data
filename_load = 'demand_train_set4.csv'

df_load = pd.read_csv(filename_load)
print(df_load.isnull().any())
load_data = np.asarray(df_load['demand_MW'])
load_hour = np.zeros((len(Hour),1))
j=0
for i in range(0,len(load_data),2):
    load_hour[j,0]= (load_data[i] + load_data[i+1])/2
    j+=1

data = np.hstack((Hour,weather_data,solar_hour,load_hour))
Names = np.hstack((['Hour'],attributeNames,attributeNames2,'demand_MW'))
dfDict = dict(zip(Names,data.T))
df=pd.DataFrame(dfDict)

df.to_excel('data_solar_prediction.xlsx',index=False)
