# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 05:09:34 2021

@author: M
"""

import streamlit as st
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as pl


st.title('Linear regions of MLP model (i.e., expresstive power measure)')
st.markdown('MLP or model is a powerful model, that has **many regions**.')

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,active = 'relu'):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''

        super(MLP, self).__init__()

        
        self.num_layers = num_layers
        self.active = active
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
            st.write('number of layers should be **positive!**')
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                
            self.linears.append(nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                if self.active == 'relu':
                    h = F.relu(self.linears[layer](h))
                elif self.active == 'sigmoid':
                    h = F.sigmoid(self.linears[layer](h))  
                else:
                    h = F.tanh(self.linears[layer](h))
                    
            return self.linears[self.num_layers - 1](h)

st.sidebar.markdown('# Setting!')
st.sidebar.markdown('### What is your favorite activation function?')
f = st.sidebar.radio('activations',('relu', 'sigmoid', 'tanh'))
n = st.sidebar.slider('Number of points?', 1000, 1000000, 1000)
l = st.sidebar.slider('Number of layers?',1,10,3)
h = st.sidebar.slider('Dim of hidden layer?',1,100,10)

size=50
inputs = npr.rand(n,2)* size -(size/2) 

st.write('**Generated DATA**')
st.dataframe(inputs)

npr.seed(10)
model = MLP(l, 2, h, 1 , active= f)

outputs = model(torch.tensor(inputs, dtype = torch.float))
outputs = outputs.detach().numpy()

fig, ax = pl.subplots()

pl.scatter(inputs[:,0],inputs[:,1],c=outputs[:,0] , cmap='rainbow' , linewidths= 0)
pl.xlim(-(size / 2), size / 2)
pl.ylim(-(size / 2), size / 2)   

st.pyplot(fig) 
       
        
        
        
        
        
        
        