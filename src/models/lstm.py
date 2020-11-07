#references-
#https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
#https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4
#https://everyd-ai.com/2018/07/15/long-short-term-memory-lstm-and-how-to-build-one-from-scratch-with-tensorflow/
#https://towardsdatascience.com/forward-and-backpropagation-in-grus-derived-deep-learning-5764f374f3f5
#Class Notes

#Note, using the same weight here for both h(t-1) and x(t) instead of seperate weights

import tensorflow as tf
import numpy as np
import math
import random


def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1./(1+np.exp(-x))

def sig_diff(x):
    return x*(1-x)

def tanh_diff(x):
    return (1. - (x**2))

def init_wt(n,joinlen):
    np.random.seed(13)
    return np.random.rand(n,joinlen)

def init_bias(n):
    np.random.seed(13)
    return np.random.rand(n)

class LossLayer:

    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

class LSTMParamInit:
    def __init__(self,cell_count,x_dim):
        self.cell_count = cell_count
        self.x_dim = x_dim
        joinlen = cell_count + x_dim
        
        #init weight matrices
        self.wf = init_wt(cell_count,joinlen)
        self.wi = init_wt(cell_count,joinlen)
        self.wg = init_wt(cell_count,joinlen)
        self.wo = init_wt(cell_count,joinlen)
        
        #init bias vecs
        self.bf = init_bias(cell_count) 
        self.bi = init_bias(cell_count) 
        self.bg = init_bias(cell_count) 
        self.bo = init_bias(cell_count) 
        
        #init derivative matrices
        self.wf_diff = np.zeros((cell_count,joinlen)) 
        self.wi_diff = np.zeros((cell_count,joinlen))
        self.wg_diff = np.zeros((cell_count,joinlen))
        self.wo_diff = np.zeros((cell_count,joinlen))
        self.bf_diff = np.zeros(cell_count) 
        self.bi_diff = np.zeros(cell_count) 
        self.bg_diff = np.zeros(cell_count) 
        self.bo_diff = np.zeros(cell_count) 
    
    def update(self,lr=0.01):
        self.wf -= lr * self.wf_diff
        self.wi -= lr * self.wi_diff
        self.wg -= lr * self.wg_diff
        self.wo -= lr * self.wo_diff
        
        self.bf -= lr * self.bf_diff
        self.bi -= lr * self.bi_diff
        self.bg -= lr * self.bg_diff
        self.bo -= lr * self.bo_diff
        
        # reset derivatives to zero (every loop)
        self.wf_diff = np.zeros_like(self.wf)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wg_diff = np.zeros_like(self.wg) 
        self.wo_diff = np.zeros_like(self.wo) 
        
        self.bf_diff = np.zeros_like(self.bf)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bg_diff = np.zeros_like(self.bg) 
        self.bo_diff = np.zeros_like(self.bo) 
    
class LSTMStateInit:
    def __init__(self, cell_count, x_dim):
        self.f = np.zeros(cell_count)
        self.i = np.zeros(cell_count)
        self.g = np.zeros(cell_count)
        self.o = np.zeros(cell_count)
        self.c = np.zeros(cell_count)
        self.h = np.zeros(cell_count)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_c = np.zeros_like(self.c)

class LSTMNode:
    def __init__(self,param,state):
        self.state = state
        self.param = param
        self.xc = None
    
    def cellinit(self,x,c_prev=None,h_prev=None):
        
        if(c_prev is None):
            c_prev = np.zeros_like(self.state.c)
        if(h_prev is None):
            h_prev = np.zeros_like(self.state.h)
        
        self.c_prev = c_prev
        self.h_prev = h_prev
        
        #concat h and x
        xc = np.hstack((x,  h_prev))
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.g = tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        
        self.state.c= self.state.g * self.state.i + c_prev * self.state.f
        self.state.h = self.state.o * tanh(self.state.c)
        
        self.xc = xc
    
    def errcalc(self,err_h,err_c):
        dc = self.state.o * err_h + err_c
        do = self.state.c * err_h
        di = self.state.g * dc
        dg = self.state.i * dc
        df = self.c_prev * dc
        
        di_input = sig_diff(self.state.i) * di 
        df_input = sig_diff(self.state.f) * df 
        do_input = sig_diff(self.state.o) * do 
        dg_input = tanh_diff(self.state.g) * dg

        # diffs wrt inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       

        # compute diffs
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save diffs
        self.state.bottom_diff_c = dc * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LSTMNet:
    def __init__(self,param):
        self.param = param
        self.nodelist = list()
        self.x_list = []
    
    def ylist(self,y_list,losslayer):
        
        #backprop
        index = len(self.x_list) - 1
        loss = losslayer.loss(self.nodelist[index].state.h, y_list[index])
        diff_h = losslayer.bottom_diff(self.nodelist[index].state.h, y_list[index])
        diff_c = np.zeros(self.param.cell_count)
        self.nodelist[index].errcalc(diff_h, diff_c)
        index-=1
        
        # we also propagate error along constant error carousel using diff_c (according to src 2)
        while index >= 0:
            loss += losslayer.loss(self.nodelist[index].state.h, y_list[index])
            diff_h = losslayer.bottom_diff(self.nodelist[index].state.h, y_list[index])
            diff_h += self.nodelist[index + 1].state.bottom_diff_h
            diff_c = self.nodelist[index + 1].state.bottom_diff_c
            self.nodelist[index].errcalc(diff_h, diff_c)
            index -= 1 
        return loss
    
    def x_list_clear(self):
        self.x_list = []
    
    def x_list_add(self,x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.nodelist):
            lstm_state = LSTMStateInit(self.param.cell_count, self.param.x_dim)
            self.nodelist.append(LSTMNode(self.param, lstm_state))
        
        index = len(self.x_list) - 1
        if index == 0:
            # no recurrent inputs yet
            self.nodelist[index].cellinit(x)
        else:
            c_prev = self.nodelist[index - 1].state.c
            h_prev = self.nodelist[index - 1].state.h
            self.nodelist[index].cellinit(x, c_prev, h_prev)