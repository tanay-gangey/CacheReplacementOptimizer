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

class GRUParamInit:
    def __init__(self,cell_count,x_dim):
        self.cell_count = cell_count
        self.x_dim = x_dim
        joinlen = cell_count + x_dim
        
        #init weight matrices
        self.wr = init_wt(cell_count,joinlen)
        self.wz = init_wt(cell_count,joinlen)
        self.wg = init_wt(cell_count,joinlen)
        
        #init bias vecs
        self.br = init_bias(cell_count) 
        self.bz = init_bias(cell_count) 
        self.bg = init_bias(cell_count) 
        
        #init derivative matrices
        self.wr_diff = np.zeros((cell_count,joinlen)) 
        self.wz_diff = np.zeros((cell_count,joinlen))
        self.wg_diff = np.zeros((cell_count,joinlen))
        self.br_diff = np.zeros(cell_count) 
        self.bz_diff = np.zeros(cell_count) 
        self.bg_diff = np.zeros(cell_count) 
        
    def update(self,lr=0.01):
        self.wr -= lr * self.wr_diff
        self.wz -= lr * self.wz_diff
        self.wg -= lr * self.wg_diff
        
        self.br -= lr * self.br_diff
        self.bz -= lr * self.bz_diff
        self.bg -= lr * self.bg_diff
        
        # reset derivatives to zero (every loop)
        self.wr_diff = np.zeros_like(self.wr)
        self.wz_diff = np.zeros_like(self.wz) 
        self.wg_diff = np.zeros_like(self.wg) 
        
        self.br_diff = np.zeros_like(self.br)
        self.bz_diff = np.zeros_like(self.bz) 
        self.bg_diff = np.zeros_like(self.bg) 
        
class GRUStateInit:
    def __init__(self, cell_count, x_dim):
        self.r = np.zeros(cell_count)
        self.z = np.zeros(cell_count)
        self.g = np.zeros(cell_count)
        self.h = np.zeros(cell_count)
        self.bottom_diff_h = np.zeros_like(self.h)

class GRUNode:
    def __init__(self,param,state):
        self.state = state
        self.param = param
        self.xc = None
    
    def cellinit(self,x,h_prev=None):
        
        if(h_prev is None):
            h_prev = np.zeros_like(self.state.h)
        
        self.h_prev = h_prev
        
        #concat h and x
        xc = np.hstack((x,  h_prev))
        self.state.r = sigmoid(np.dot(self.param.wr, xc) + self.param.br)
        self.state.z = sigmoid(np.dot(self.param.wz, xc) + self.param.bz)
        #self.state.g = tanh(np.dot(self.param.wg1, (self.state.r*h_prev)) + np.dot(self.param.wg2,x) + self.param.bg)
        #shape issues above ^^^
        self.state.g = tanh(np.dot(self.param.wg, xc) + self.param.bg)
        #simplified write gate above^^^
        self.state.h= self.state.g * self.state.z + h_prev * (1-self.state.z)
        
        self.xc = xc
    
    def errcalc(self,err_h):
        dh = err_h
        d3 = self.state.g * dh #i-3
        d1 = self.state.z * dh #g-1
        d2 = self.h_prev * dh #f-2
        
        dz_input = sig_diff(self.state.z) * d3 
        dr_input = sig_diff(self.state.r) * d2 
        dg_input = tanh_diff(self.state.g) * d1

        # diffs wrt inputs
        self.param.wz_diff += np.outer(dz_input, self.xc)
        self.param.wr_diff += np.outer(dr_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bz_diff += dz_input
        self.param.br_diff += dr_input       
        self.param.bg_diff += dg_input       

        # compute diffs
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wz.T, dz_input)
        dxc += np.dot(self.param.wr.T, dr_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save diffs
        #self.state.bottom_diff_c = dh * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

class GRUNet:
    def __init__(self,param):
        self.param = param
        self.nodelist = list()
        self.x_list = []
    
    def ylist(self,y_list,losslayer):
        
        #backprop
        index = len(self.x_list) - 1
        loss = losslayer.loss(self.nodelist[index].state.h, y_list[index])
        diff_h = losslayer.bottom_diff(self.nodelist[index].state.h, y_list[index])
        #diff_c = np.zeros(self.param.cell_count)
        self.nodelist[index].errcalc(diff_h)
        index-=1
        
        # we also propagate error along constant error carousel using diff_c (according to src 2)
        while index >= 0:
            loss += losslayer.loss(self.nodelist[index].state.h, y_list[index])
            diff_h = losslayer.bottom_diff(self.nodelist[index].state.h, y_list[index])
            diff_h += self.nodelist[index + 1].state.bottom_diff_h
            #diff_c = self.nodelist[index + 1].state.bottom_diff_c
            self.nodelist[index].errcalc(diff_h)
            index -= 1 
        return loss
    
    def x_list_clear(self):
        self.x_list = []
    
    def x_list_add(self,x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.nodelist):
            gru_state = GRUStateInit(self.param.cell_count, self.param.x_dim)
            self.nodelist.append(GRUNode(self.param, gru_state))
        
        index = len(self.x_list) - 1
        if index == 0:
            # no recurrent inputs yet
            self.nodelist[index].cellinit(x)
        else:
            h_prev = self.nodelist[index - 1].state.h
            self.nodelist[index].cellinit(x, h_prev)