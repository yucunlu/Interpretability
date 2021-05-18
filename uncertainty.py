# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:03:44 2021

@author: Lycanthrope
"""
import pandas as pd
import numpy as np
import copy
import tensorflow as tf
from matplotlib import pyplot as plt
import tikzplotlib
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


feature_names =  [('Hour'+ str(x)) for x in range(24)]+list(attributenames)+['Hour_temp','Hour_gen','Day_gen','Hour_load','Gen','Load','gen-load']
dfDict = dict(zip(feature_names,dataset_final.T))
df=pd.DataFrame(dfDict)

# load = 0 hours are deemed as abnormal hours and are excluded
#Remaining hours are from 7:00 - 18:00
df_clear = df.drop(df[(df['Hour0']==1) | (df['Hour1']==1) | (df['Hour2']==1) | (df['Hour3']==1) |
                      (df['Hour4']==1) | (df['Hour5']==1) | (df['Hour6']==1) |
                      (df['Hour18']==1)| (df['Hour19']==1) | (df['Hour20']==1) |
                      (df['Hour21']==1) | (df['Hour22']==1) | (df['Hour23']==1) | (df['Load']==0) |(df['Hour_load']==0)
                      ].index)

df_clear = df_clear.drop([('Hour'+ str(x)) for x in range(7)], axis=1)
df_clear = df_clear.drop([('Hour'+ str(x)) for x in range(18,24)], axis=1)

feature_names =  [('Hour'+ str(x)) for x in range(7,18)] +['temp_forecast','Irra_forecast','Panel_temp','Hour_gen','Day_gen','Hour_load']

feature_names = np.asarray(feature_names)
data = np.asarray(df_clear)


np.random.seed(10)
per = np.random.permutation(data.shape[0])		#Row number after shuffling

X = data[:,:-1]		#Obtain data after shuffling
y = data[:,-1]

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

X_train_solar=copy.deepcopy(X_train[:,:-1])
y_train_solar=copy.deepcopy(X[:train_size,-2])
X_valid_solar=copy.deepcopy(X_valid[:,:-1])
y_valid_solar=copy.deepcopy(X[train_size:,-2])


'''
#Predicting Epistemic Uncertainty (MC dropout)

Input = tf.keras.Input(shape=(16,))
x = tf.keras.layers.Dense(50, activation='relu')(Input)
x = tf.keras.layers.Dropout(0.1)(x, training=True)
x =  tf.keras.layers.Dense(30, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x, training=True)
x =  tf.keras.layers.Dense(10, activation='relu')(x)
Output =  tf.keras.layers.Dense(1)(x)

model_MC = tf.keras.models.Model(Input, Output)

checkpoint_filepath = 'MCdropout/checkpoint'

model_MC.load_weights(checkpoint_filepath)

predictions = []
for _ in range(500):
    predictions += [model_MC.predict(X_valid_solar)]    
epistemic_mean, epistemic_std = np.mean(np.array(predictions), axis=0), np.std(np.array(predictions), axis=0)


rmse = np.sqrt(np.sum((y_valid_solar-epistemic_mean)**2)/len(y_valid_solar))
print(rmse)
'''
'''
from tensorflow.keras import backend as K
#Predicting Aleatoric Uncertainty
def aleatoric_loss(y_true, y_pred):
    N = y_true.shape[0]
    se = K.pow((y_true[:,0]-y_pred[:,0]),2)
    inv_std = K.exp(-y_pred[:,1])
    mse = K.mean(inv_std*se)
    reg = K.mean(y_pred[:,1])
    return 0.5*(mse + reg)

y_train_reshaped = np.vstack([y_train_solar, np.zeros(y_train_solar.shape)]).T

model_aleatoric  = tf.keras.models.Sequential([
            tf.keras.layers.Dense(50, input_shape=[16], activation='relu',kernel_regularizer='l2'),
            tf.keras.layers.Dense(30, activation='relu',kernel_regularizer='l2'),
            tf.keras.layers.Dense(10, activation='relu',kernel_regularizer='l2'),
            tf.keras.layers.Dense(2),
            ])

model_aleatoric.compile(loss=aleatoric_loss,optimizer=tf.keras.optimizers.Adam(lr=1e-4),metrics=['mae'])
history=model_aleatoric.fit(X_train_solar, y_train_reshaped, epochs=600)


def predictor(model, 
              X_test, T=100):
    probs = []
    for _ in range(T):
        probs += [model.predict(X_test,verbose=0)]
    return probs

p = np.array(predictor(model_aleatoric, X_valid_solar, T=1))

aleatoric_mean, std = np.mean(p[:,:,0], axis=0), np.std(p[:,:,0], axis=0)
aleatoric_std = np.exp(0.5*np.mean(p[:,:,1], axis=0)).reshape(-1,1)

uncertainty = np.sqrt(aleatoric_std**2+epistemic_std**2)

from scipy.stats import norm

ub_95 = epistemic_mean+norm.ppf(0.95)*uncertainty
lb_95 = epistemic_mean-norm.ppf(0.95)*uncertainty


ub_9 = epistemic_mean+norm.ppf(0.9)*uncertainty
lb_9 = epistemic_mean-norm.ppf(0.9)*uncertainty


ub_8 = epistemic_mean+norm.ppf(0.8)*uncertainty
lb_8 = epistemic_mean-norm.ppf(0.8)*uncertainty



in_the_range=(y_valid_solar<ub_95)&(y_valid_solar>lb_95)
sum(in_the_range)[0]/1063

x_axis=range(50)
y_axis=epistemic_mean[50:100,0]
fig, ax = plt.subplots()
ax.plot(x_axis,y_axis,'o-',label='mean',color='k',alpha=0.3)
ax.plot(x_axis,y_valid_solar[50:100,0],'*-',label='Real',color='r',alpha=0.3)
ax.fill_between(x_axis, lb_95[50:100,0], ub_95[50:100,0], color='b', alpha=.2,label='95%')
ax.fill_between(x_axis, lb_9[50:100,0], ub_9[50:100,0], color='b', alpha=.3,label='90%')
ax.fill_between(x_axis, lb_8[50:100,0], ub_8[50:100,0], color='b', alpha=.5,label='80%')
ax.yaxis.set_ticks([0,500,1000,1500,2000,2500])  
ax.legend(borderpad=0.1,labelspacing=0.1,loc='upper left',prop = {'size':7})

'''


#Bayesian Neural networks
import tensorflow_probability as tfp

batch_size = 32
batch_num = train_size/batch_size
kl_loss_weight = 1.0 / train_size
  
# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def  posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tfp.layers.DenseVariational(units=50,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=30,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=10,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=1+1,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable,
                                kl_weight=kl_loss_weight,
                                ),
     tfp.layers.DistributionLambda(
      lambda t: tfp.distributions.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))
    ])





checkpoint_filepath = 'DNN_BNN_v2/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='mse',
    mode='min',
    save_best_only=True)


negloglik = lambda y, rv_y: -rv_y.log_prob(y)

model.compile(loss=negloglik,optimizer=tf.keras.optimizers.Adam(lr=1e-3),metrics=['mse'])
model.fit(X_train_solar, y_train_solar, epochs=600,callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)

yhats = [model(X_valid_solar) for _ in range(100)]
avgm = np.zeros_like(X_valid_solar[..., 0])


plt.figure(figsize=(20,10))
for i, yhat in enumerate(yhats):
  m = np.squeeze(yhat.mean())
  s = np.squeeze(yhat.stddev())
  
  if i < 10:
    plt.plot(range(45), m[5:50], 'r-o',label='Epistemic uc' if i == 0 else None, linewidth=0.5)
    plt.plot(range(45), m[5:50] + 2 * s[5:50], 'g', linewidth=0.5, label='aleatoric uc(95% CI)' if i == 0 else None);
    plt.plot(range(45), m[5:50] - 2 * s[5:50], 'g', linewidth=0.5 if i == 0 else None);
  
  avgm += m


avgm = (avgm/len(yhats))
plt.legend(loc='upper right',fontsize=25)
plt.tick_params(labelsize=25)
plt.xlabel('Hour',fontsize=25)
plt.ylabel('Solar power prediction (KWh)',fontsize=25)
plt.savefig("plots/BNN.pdf")

rmse = np.sqrt(np.sum((y_valid_solar-avgm)**2)/len(y_valid_solar))

m = np.zeros((1063,100))
s = np.zeros((1063,100))

for i in range(100):
    m[:,i] = np.squeeze(yhats[i].mean())
    s[:,i] = np.squeeze(yhats[i].stddev())

aleatoric_uc = np.mean(s,axis=1)
epistemic_uc = np.std(m,axis=1)

uncertainty = np.sqrt(aleatoric_uc**2 + epistemic_uc**2)



plt.figure(figsize=(20,10))
plt.plot(range(45),avgm[5:50],'r-o',label='Predicted')
plt.plot(range(45),y_valid_solar[5:50],'g-x',label='Real')
plt.fill_between(range(45), avgm[5:50]-1.96*uncertainty[5:50], avgm[5:50]+1.96*uncertainty[5:50], color='silver', alpha=1,label='95% CI')
plt.legend(loc='upper right',fontsize=22)
plt.tick_params(labelsize=25)
plt.xlabel('Hour',fontsize=25)
plt.ylabel('Solar power prediction (KWh)',fontsize=25)
plt.savefig("plots/PF.pdf")


predictions = []
for _ in range(500):
    predictions += [model.predict(X_valid_solar)]    
prediction, epistemic_std = np.mean(np.array(predictions), axis=0), np.std(np.array(predictions), axis=0)


rmse = np.sqrt(np.sum((y_valid_solar.reshape(-1,1)-prediction)**2)/len(y_valid_solar))

'''
@tf.function
def elbo_loss(labels, logits):
    loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    loss_kl = tf.keras.losses.KLD(labels, logits)
    loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
    return loss
'''

#interpretability
#deeplift
'''
import shap
X_base_solar = np.zeros((1,16))

Y_base_solar = model.predict(X_base_solar)
for _ in range(500):
    predictions += [model.predict(X_base_solar)]    
Y_base_solar = np.mean(np.array(predictions), axis=0)



X_base_solar = X_base_solar.astype('float32')
X_valid_solar = X_valid_solar.astype('float32')
background_solar = X_base_solar


shap_values_solar = np.zeros((1063,16))
for i in range(100):
    e_solar = shap.DeepExplainer(model, background_solar)
    shap_values_solar = shap_values_solar+e_solar.shap_values(X_valid_solar,check_additivity=False)[0]

shap_values_solar = shap_values_solar/100
    
heatmap = np.abs(shap_values_solar)
argmax = np.argmax(heatmap,axis=1)
unique, counts = np.unique(argmax, return_counts=True)
unique_str = feature_names[unique]
counts = counts/1063
deeplift_solar = dict(zip(unique_str, counts))
deeplift_average_solar = np.average(heatmap,axis=0)

shap.force_plot(e_solar.expected_value.numpy(), shap_values_solar[12], df_solar.iloc[3,:],matplotlib=True)

y_BNN=np.sum(shap_values_solar,axis=1) + Y_base_solar[0][0]

error = np.sum(np.abs(y_BNN.reshape(-1,1)-prediction))/len(prediction)
'''