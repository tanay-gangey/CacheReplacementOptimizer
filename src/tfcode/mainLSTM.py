import numpy as np
import pandas as pd
from lstm import LSTMParamInit, LSTMNet, LossLayer
from sklearn.preprocessing import minmax_scale


def split(data, size):
    sequences, y = list(), list()
    for i in range(len(data)):
        end = i + size
        if end >= len(data):
            break
        sequences.append(np.array(data[i:end],dtype=np.float64))
        y.append(data[end])
    return sequences, y

def makedata(n):
    df = pd.read_csv("../../../../blkIO.txt", sep=' ',header = None)
    df.columns = ['timestamp','pid','pname','blockNo', 'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
    df = df.drop(['pid', 'pname', 'blockSize', 'bdMajor', 'bdMinor', 'hash'], axis=1)
    readsAndWrites=df['blockNo'].tolist()
    readsAndWrites = minmax_scale(readsAndWrites,feature_range=(0,256))
    x, y = split(readsAndWrites[:int(0.05*len(readsAndWrites))], n)
    print(df.head())
    return x,y
    

def main():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    
    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 1000
    lstm_param = LSTMParamInit(mem_cell_ct, x_dim)
    lstm_net = LSTMNet(lstm_param)
    print("Before creating data")
    input_val_arr,y_list = makedata(x_dim)
    print("After creating data",len(input_val_arr),len(y_list),len(input_val_arr[0]))
    for epoch in range(10):
        print("epoch", "%2s" % str(epoch), end=": ")
        for index in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[index])

            
        loss = lstm_net.ylist(y_list, LossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.update(lr=1)
        lstm_net.x_list_clear()
    print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.nodelist[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")

if __name__ == "__main__":
    main()
