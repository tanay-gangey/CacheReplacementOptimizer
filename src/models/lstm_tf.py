import tensorflow as tf #using tf version == 1.14
from tensorflow.contrib import rnn

#clear the graph
tf.compat.v1.reset_default_graph()

class LSTM_NET:

    def __init__(self, x, num_hidden, num_classes, timesteps):
        # data shape to match `rnn` function requirements
        # data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        self.x = tf.unstack(x, timesteps, 1)

        # Define weights and biases for hidden and output dense layers
        self.weights = {
            'hidden' : tf.Variable(tf.random_normal([num_hidden, num_hidden])),
            'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        }
        self.biases = {
            'hidden' : tf.Variable(tf.random_normal([num_hidden])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }


        # single layer with multiple lstm cells
        # self.lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1, state_is_tuple= True)
        
        #creating multilayer rnnCells as of now 3 layers
        self.rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden),rnn.BasicLSTMCell(num_hidden),rnn.BasicLSTMCell(num_hidden)])

    def build_architecture_forward_pass(self):

        outputs, states = rnn.static_rnn(self.rnn_cell, self.x, dtype=tf.float32)

        #adding two dense layers to increase the non linearity in predictions along with relu activatoins
        hidden_out = tf.nn.tanh(tf.add(tf.matmul(outputs[-1], self.weights['hidden']), self.biases['hidden']))
        hidden_out = tf.nn.relu(tf.add(tf.matmul(hidden_out, self.weights['hidden']), self.biases['hidden']))
        #using a linear activation for predictions
        output = tf.add(tf.matmul(hidden_out, self.weights['out']), self.biases['out'])
        
        return output

    