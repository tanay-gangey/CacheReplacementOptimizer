from lstm_tf import *
import numpy as np
import tensorflow as tf

# Training Parameters
learning_rate = 0.0001
training_steps = 100000
batch_size = 1000
display_step = 100

# Network Parameters
num_input = 999 # data input
timesteps = 1 # timesteps
num_hidden = 100 # hidden layer num of features
num_classes = 1 # output class length, 1, since were predicting only one number

def run_the_model(x_train, y_train, x_test, y_test):
        # tf Graph input and output, placeholders
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])

        network = LSTM_NET(X, num_hidden, num_classes, timesteps)

        #build the architecture along with which the forward pass is kept ready
        logits = network.build_architecture_forward_pass()
        prediction = logits

        # Define loss and optimizer, mse and Adam resp
        loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model, with soft accuracy metric
        correct_pred = tf.equal(tf.math.round(prediction), tf.math.round(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)
            #to load the batches in a round robin fashion
            i = 0

            #training steps is crucial for deciding how long should the model run.
            for step in range(1, training_steps+1):

                # getting the next batch of the already shuffled data along with taking care of an extra dimension thats required
                batch_x, batch_y = np.array(x_train[i : i + batch_size])[:, None], np.array(y_train[i : i + batch_size])[:, None]
                
                # Reshape data to get num_inputs per timestep
                batch_x = batch_x.reshape(batch_size, timesteps, num_input)
                
                # Time to run the backpropogation!
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc, pred = sess.run([loss_op, accuracy, prediction], feed_dict={X: batch_x,
                                                                        Y: batch_y})

                    #uncomment this to see the predicted and actual values while training
                    # print("predicted value", pred[:5], "actual", batch_y[:5])
                    print("Step " + str(step) + ", Batch Loss= " + \
                        "{:.4f}".format(loss) + ", Training Accuracy= " + \
                        "{:.3f}".format(acc))
                if(i == len(x_train) - batch_size):
                    i = 0
                else:
                    i += batch_size
            print("Optimization Finished!")

            x_test = x_test.reshape(-1, timesteps, num_input)
            y_test = y_test.reshape(len(y_test),num_classes)

            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
            print(sess.run(prediction, feed_dict = {X: x_test}))
            print(y_test)


def main():
    #load the preprocessed data, these files contain the training and testing instances after shuffling.
    
    sequences = np.load("../data/sequences_data.npy", allow_pickle=True)
    labels = np.load("../data/label_data.npy", allow_pickle=True)
    
    x_train, y_train = sequences[0], labels[0]
    x_test, y_test = sequences[1], labels[1]
    
    run_the_model(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()



