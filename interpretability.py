# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:03:44 2021

@author: Lycanthrope
"""
import pandas as pd
import numpy as np
import copy
import tensorflow as tf

filename = 'data_solar_prediction.xlsx'

df = pd.read_excel(filename)

Hours=np.asarray(df['Hour'])
Hours = Hours[:,np.newaxis]
#one-hot encoder
K = Hours.max()
Hour_encoding = np.zeros((Hours.size, K+1))
Hour_encoding[np.arange(Hours.size), Hours[:,0]] = 1

attributenames = ['temp_location2','solar_location2']
data_weather=np.asarray(df[attributenames])

attributenames_solar  = np.asarray(df.columns[13:16])
data_solar=np.asarray(df[attributenames_solar])

data_solar[:,2] = data_solar[:,2]/5*2000

Hour_ahead_info = np.ones((len(data_solar),2))*-1
Hour_ahead_info[1:,:] = data_solar[:-1,1:]

Day_ahead_info = np.ones((len(data_solar),1))*-1
Day_ahead_info[24:,0] = data_solar[:-24,2]

solar_power = copy.deepcopy(data_solar[:,2])
solar_power = solar_power[:,np.newaxis]


attributenames_load  = np.asarray(df.columns[16])
data_load=np.asarray(df[attributenames_load]).reshape(-1,1)
data_load = data_load/5*1000

hour_ahead_load = np.ones((len(data_load),1))*-1
hour_ahead_load[1:,0] = data_load[:-1,0]


gen_load= (solar_power>data_load)*1

dataset = np.hstack((Hour_encoding,data_weather,Hour_ahead_info,Day_ahead_info,hour_ahead_load,solar_power,data_load,gen_load))

dataset = dataset[24:,:]

print(sum(np.isnan(dataset).any(axis=1)))
dataset_final =dataset[~np.isnan(dataset).any(axis=1), :]


feature_names =  [('Hour'+ str(x)) for x in range(24)]+['STemp','Irra','PTemp','HPow','DPow','HLoad','Pow','Load','Pow-load']
dfDict = dict(zip(feature_names,dataset_final.T))
df=pd.DataFrame(dfDict)

# load = 0 hours are deemed as abnormal hours and are excluded
#Remaining hours are from 7:00 - 18:00
df_clear = df.drop(df[(df['Hour0']==1) | (df['Hour1']==1) | (df['Hour2']==1) | (df['Hour3']==1) |
                      (df['Hour4']==1) | (df['Hour5']==1) | (df['Hour6']==1) |
                      (df['Hour18']==1)| (df['Hour19']==1) | (df['Hour20']==1) |
                      (df['Hour21']==1) | (df['Hour22']==1) | (df['Hour23']==1) | (df['Load']==0) |(df['HLoad']==0)
                      ].index)

df_clear = df_clear.drop([('Hour'+ str(x)) for x in range(7)], axis=1)
df_clear = df_clear.drop([('Hour'+ str(x)) for x in range(18,24)], axis=1)

feature_names =  [('Hour'+ str(x)) for x in range(7,18)] +['STemp','Irra','PTemp','HPow','DPow','HLoad']

feature_names = np.asarray(feature_names)
data = np.asarray(df_clear)


np.random.seed(10)
per = np.random.permutation(data.shape[0])		#Row number after shuffling

X = data[per,:-1]		#Obtain data after shuffling
y = data[per,-1]

#Generate training and validation set

train_size=int(len(y)*0.9)
X_train=copy.deepcopy(X[:train_size,:-2])
y_train=copy.deepcopy(y[:train_size])
X_valid=copy.deepcopy(X[train_size:,:-2])
y_valid=copy.deepcopy(y[train_size:])

X_valid_origin = copy.deepcopy(X[train_size:])

#Normalize input features


'''
mu = np.mean(X_train[:,11:],axis=0)
sigma =  np.std(X_train[:,11:],axis=0)

X_train[:,11:] = (X_train[:,11:] -mu)/sigma
X_valid[:,11:] = (X_valid[:,11:] -mu)/sigma
'''

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, input_shape=[17], activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(30, activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(10, activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()

checkpoint_filepath = 'DNN_loadcontrol_v2/checkpoint'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.Adam(lr=1e-4),metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=600, batch_size = 32,validation_data=(X_valid, y_valid),callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)
ys=model.predict(X_valid)
y_valid=y_valid.reshape(-1,1)

accuracy = np.sum(np.round(ys) == y_valid)/len(ys)

TP=np.sum((np.round(ys)==1) & (y_valid==1))
FN=np.sum((np.round(ys)==0) & (y_valid==1))
FP=np.sum((np.round(ys)==1) & (y_valid==0))
TN=np.sum((np.round(ys)==0) & (y_valid==0))



#Gradient
inp = tf.Variable(X_valid, dtype=tf.float32)

with tf.GradientTape() as tape:
    preds = model(inp)

grads = tape.gradient(preds, inp)

gradients = grads.numpy()

heatmap = np.abs(gradients)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
grad = dict(zip(unique_str, counts))
g_average = np.average(heatmap,axis=0)


#Gradient * Input
grad_input = gradients * X_valid

heatmap = np.abs(grad_input)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
gradients_input = dict(zip(unique_str, counts))
gi_average = np.average(heatmap,axis=0)


X_base = np.zeros((1,17))
Y_base = model.predict(X_base)

'''
X_base = np.mean(X_train,axis=0).reshape(1,-1)
X_base[0,0:11]=0
Y_base = model.predict(X_base)
'''
from path_explain import PathExplainerTF


#Integrated gradients
explainer = PathExplainerTF(model)

X_base = X_base.astype('float32')
X_valid = X_valid.astype('float32')
X_train = X_train.astype('float32')

mean = np.zeros((10,1))
best= np.zeros((10,1))
worst = np.zeros((10,1))
pct10 = np.zeros((10,1))
pct90 = np.zeros((10,1))

steps = [10,20,40,100,200,300,400,500,600,800]

for i in range(10):
    attributions_ig = explainer.attributions(inputs=X_valid,baseline=X_base,num_samples=steps[i],use_expectation=False,)
    completeness = np.sum(attributions_ig[0],axis=1)
    diff = completeness + Y_base[0][0] * np.ones((1063,))
    path_integral = np.abs(diff - ys[:,0])
    mean[i] = np.mean(path_integral)
    best[i] = np.min(path_integral)
    worst[i] = np.max(path_integral)
    pct10[i] = np.percentile(path_integral, 10)
    pct90[i] = np.percentile(path_integral, 90)
    

import matplotlib.pyplot as plt
import tikzplotlib


plt.figure()
plt.plot(steps,mean,'-o',label='Mean',color='cornflowerblue',alpha=1)
plt.fill_between(steps, worst[:,0], best[:,0], color='silver', alpha=1,label='Worst/best case')
plt.fill_between(steps, pct10[:,0], pct90[:,0], color='lightskyblue', alpha=1,label='[10%,90%] -percentile')
plt.legend(borderpad=0.2,labelspacing=0.2,loc='upper right',prop = {'size':8})
plt.title('Completeness convergence of Integrated Gradients')
plt.xlabel('Number of steps') 
plt.ylabel('Completeness difference')
plt.yscale('log')
plt.ylim([0.0001, 1])

plt.savefig("completeness1.pdf")


heatmap = np.abs(attributions_ig[0])
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
integrated_gradiants = dict(zip(unique_str, counts))
ig_average = np.average(heatmap,axis=0)


#Expected gradients

X_background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]
Y_background = model.predict(X_background)

for i in range(10):
    attributions_eg = explainer.attributions(inputs=X_valid,baseline=X_background,num_samples=steps[i],use_expectation=True)
    completeness = np.sum(attributions_eg[0],axis=1)
    diff = completeness + np.mean(Y_background) * np.ones((1063,))
    path_integral = np.abs(diff - ys[:,0])
    mean[i] = np.mean(path_integral)
    best[i] = np.min(path_integral)
    worst[i] = np.max(path_integral)
    pct10[i] = np.percentile(path_integral, 10)
    pct90[i] = np.percentile(path_integral, 90)
    
plt.figure()
plt.plot(steps,mean,'-o',label='Mean',color='cornflowerblue',alpha=1)
plt.fill_between(steps, worst[:,0], best[:,0], color='silver', alpha=1,label='Worst/best case')
plt.fill_between(steps, pct10[:,0], pct90[:,0], color='lightskyblue', alpha=1,label='[10%,90%] -percentile')
plt.legend(borderpad=0.2,labelspacing=0.2,loc='upper right',prop = {'size':8})
plt.title('Completeness convergence of Expected Gradients')
plt.xlabel('Number of steps') 
plt.ylabel('Completeness difference')
plt.yscale('log')
plt.ylim([0.0001, 1])

plt.savefig("plots/completeness2.pdf")

heatmap = np.abs(attributions_eg[0])
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
expected_gradiants = dict(zip(unique_str, counts))
eg_average = np.average(heatmap,axis=0)

#DeepLIFT
import shap
dfDict = dict(zip(feature_names,X_valid_origin[:,:-2].T))
df=pd.DataFrame(dfDict)
df=df.round(1)
background = X_base
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_valid)[0]

heatmap = np.abs(shap_values)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
deeplift = dict(zip(unique_str, counts))
deeplift_average = np.average(heatmap,axis=0)


#Stochastic Ablation test
'''
X_valid_ablate1 = copy.deepcopy(X_valid)
X_valid_ablate1[:,16]= np.mean(X_train[:,16])
X_valid_ablate1[:,14]=np.mean(X_train[:,14])
X_valid_ablate1[:,12]=np.mean(X_train[:,12])
ys_ablate1 = model.predict(X_valid_ablate1)
accuracy_abla1 = np.sum(np.round(ys_ablate1) == y_valid)/len(ys_ablate1 )
(accuracy-accuracy_abla1)
'''

X_train_solar=copy.deepcopy(X_train[:,:-1])
y_train_solar=copy.deepcopy(X[:train_size,-2])
X_valid_solar=copy.deepcopy(X_valid[:,:-1])
y_valid_solar=copy.deepcopy(X[train_size:,-2])



model_solar = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, input_shape=[16], activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(30, activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(10, activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(1,  activation = 'sigmoid'),
    tf.keras.layers.Lambda(lambda x: x * 2000)
])

checkpoint_filepath = 'DNN_solar_scale/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
model_solar.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=1e-3))
history=model_solar.fit(X_train_solar, y_train_solar, batch_size=32, epochs=600, validation_data=(X_valid_solar, y_valid_solar),callbacks=[model_checkpoint_callback])


model_solar.load_weights(checkpoint_filepath)
y_solar_pre=model_solar.predict(X_valid_solar)
y_valid_solar=y_valid_solar.reshape(-1,1)

rmse = np.sqrt(np.sum((y_valid_solar-y_solar_pre)**2)/len(y_valid_solar))

y_valid_load=copy.deepcopy(X_valid_origin[:,16])
y_valid_load = y_valid_load.reshape(-1,1)
ys_solar = (y_solar_pre>y_valid_load)*1
accuracy_2 = np.sum(ys_solar == y_valid)/len(ys_solar)

TP2=np.sum((ys_solar==1) & (y_valid==1))
FN2=np.sum((ys_solar==0) & (y_valid==1))
FP2=np.sum((ys_solar==1) & (y_valid==0))
TN2=np.sum((ys_solar==0) & (y_valid==0))


#Stochastic ablation test

'''
X_valid_ablate1 = copy.deepcopy(X_valid_solar)
X_valid_ablate1[:,14]= np.mean(X_train[:,14])
X_valid_ablate1[:,12]= np.mean(X_train[:,12])
X_valid_ablate1[:,13]= np.mean(X_train[:,13])
y_solar_ablate1 = model_solar.predict(X_valid_ablate1)
rmse_ablate1 = np.sqrt(np.sum((y_valid_solar-y_solar_ablate1)**2)/len(y_valid_solar))
rmse_ablate1-rmse
'''

#Gradient
inp = tf.Variable(X_valid_solar, dtype=tf.float32)

with tf.GradientTape() as tape:
    preds = model_solar(inp)

grads = tape.gradient(preds, inp)

gradients = grads.numpy()

heatmap = np.abs(gradients)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
grad = dict(zip(unique_str, counts))
g_average = np.average(heatmap,axis=0)

#Gradient * Input
grad_input = gradients * X_valid_solar

heatmap = np.abs(grad_input)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
gradients_input = dict(zip(unique_str, counts))
gi_average = np.average(heatmap,axis=0)

#DeepLIFT
dfDict_solar = dict(zip(feature_names[:-1],X_valid_origin[:,:-3].T))
df_solar=pd.DataFrame(dfDict_solar)
df_solar=df_solar.round(1)

X_base_solar = np.zeros((1,16))
Y_base_solar = model_solar.predict(X_base_solar)

'''
X_base_solar = np.mean(X_train_solar,axis=0).reshape(1,-1)
X_base_solar[0,0:11]=0
Y_base_solar = model_solar.predict(X_base_solar)
'''

X_base_solar = X_base_solar.astype('float32')
X_valid_solar = X_valid_solar.astype('float32')
background_solar = X_base_solar
e_solar = shap.DeepExplainer(model_solar, background_solar)

shap_values_solar = e_solar.shap_values(X_valid_solar)[0]

heatmap = np.abs(shap_values_solar)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
deeplift_solar = dict(zip(unique_str, counts))
deeplift_average_solar = np.average(heatmap,axis=0)

#integrated gradients
X_base_solar = np.zeros((1,16))
Y_base_solar = model_solar.predict(X_base_solar)

from path_explain import PathExplainerTF

explainer = PathExplainerTF(model_solar)

X_base_solar = X_base_solar.astype('float32')
X_valid_solar = X_valid_solar.astype('float32')
X_train_solar = X_train_solar.astype('float32')

mean = np.zeros((10,1))
best= np.zeros((10,1))
worst = np.zeros((10,1))
pct10 = np.zeros((10,1))
pct90 = np.zeros((10,1))

steps = [10,20,40,100,200,300,400,500,600,800]

for i in range(10):
    attributions_ig = explainer.attributions(inputs=X_valid_solar,baseline=X_base_solar,num_samples=steps[i],use_expectation=False,)
    completeness = np.sum(attributions_ig[0],axis=1)
    diff = completeness + Y_base_solar[0][0] * np.ones((1063,))
    path_integral = np.abs(diff - y_solar_pre[:,0])
    mean[i] = np.mean(path_integral)
    best[i] = np.min(path_integral)
    worst[i] = np.max(path_integral)
    pct10[i] = np.percentile(path_integral, 10)
    pct90[i] = np.percentile(path_integral, 90)
    

plt.figure()

plt.plot(steps,mean,'-o',label='Mean',color='cornflowerblue',alpha=1)
plt.fill_between(steps, worst[:,0], best[:,0], color='silver', alpha=1,label='Worst/best case')
plt.fill_between(steps, pct10[:,0], pct90[:,0], color='lightskyblue', alpha=1,label='[10%,90%] -percentile')
plt.legend(borderpad=0.2,labelspacing=0.2,loc='upper right',prop = {'size':8})
plt.title('Completeness convergence of Integrated Gradients')
plt.xlabel('Number of steps') 
plt.ylabel('Completeness difference')


plt.savefig("plots/completeness3.pdf")

heatmap = np.abs(attributions_ig[0])
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
integrated_gradiants = dict(zip(unique_str, counts))
ig_average = np.average(heatmap,axis=0)


#Expected gradients
X_background = X_train_solar[np.random.choice(X_train_solar.shape[0], 1000, replace=False)]
Y_background = model_solar.predict(X_background)

for i in range(10):
    attributions_eg = explainer.attributions(inputs=X_valid_solar,baseline=X_background,num_samples=steps[i],use_expectation=True)
    completeness = np.sum(attributions_eg[0],axis=1)
    diff = completeness + np.mean(Y_background) * np.ones((1063,))
    path_integral = np.abs(diff - y_solar_pre[:,0])
    mean[i] = np.mean(path_integral)
    best[i] = np.min(path_integral)
    worst[i] = np.max(path_integral)
    pct10[i] = np.percentile(path_integral, 10)
    pct90[i] = np.percentile(path_integral, 90)

plt.figure()

plt.plot(steps,mean,'-o',label='Mean',color='cornflowerblue',alpha=1)
plt.fill_between(steps, worst[:,0], best[:,0], color='silver', alpha=1,label='Worst/best case')
plt.fill_between(steps, pct10[:,0], pct90[:,0], color='lightskyblue', alpha=1,label='[10%,90%] -percentile')
plt.legend(borderpad=0.2,labelspacing=0.2,loc='upper right',prop = {'size':8})
plt.title('Completeness convergence of Expected Gradients')
plt.xlabel('Number of steps') 
plt.ylabel('Completeness difference')
plt.ylim([0, 250])

plt.savefig("plots/completeness4.pdf")

heatmap = np.abs(attributions_eg[0])
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
expected_gradiants = dict(zip(unique_str, counts))
eg_average = np.average(heatmap,axis=0)


#local cases

#True positive
#3 Real: 1180 463
shap.force_plot(e.expected_value.numpy(), shap_values[8], df.iloc[8,:],matplotlib=True,show=False)
plt.savefig('TP.pdf',bbox_inches = 'tight')
shap.force_plot(e_solar.expected_value.numpy(), shap_values_solar[8], df_solar.iloc[8,:],matplotlib=True,show=False)
plt.savefig('TP_solar.pdf',bbox_inches = 'tight')

#True negative
#4 Real: 6 881
shap.force_plot(e.expected_value.numpy(), shap_values[13], df.iloc[13,:],matplotlib=True,show=False)
plt.savefig('TN.pdf',bbox_inches = 'tight')
shap.force_plot(e_solar.expected_value.numpy(), shap_values_solar[13], df_solar.iloc[13,:],matplotlib=True,show=False)
plt.savefig('TN_solar.pdf',bbox_inches = 'tight')
#False negative
#17 Real: 784 626
shap.force_plot(e.expected_value.numpy(), shap_values[194], df.iloc[194,:],matplotlib=True,show=False)
plt.savefig('FN.pdf',bbox_inches = 'tight')
shap.force_plot(e_solar.expected_value.numpy(), shap_values_solar[194], df_solar.iloc[194,:],matplotlib=True,show=False)
plt.savefig('FN_solar.pdf',bbox_inches = 'tight')
#False positive
#71 Real: 470 555
shap.force_plot(e.expected_value.numpy(), shap_values[238], df.iloc[238,:],matplotlib=True,show=False)
plt.savefig('FP.pdf',bbox_inches = 'tight')
shap.force_plot(e_solar.expected_value.numpy(), shap_values_solar[238], df_solar.iloc[238,:],matplotlib=True,show=False)
plt.savefig('FP_solar.pdf',bbox_inches = 'tight')