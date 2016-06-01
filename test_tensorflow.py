
import tensorflow as tf
import numpy as np


rng = np.random
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# tf Graph Input
X = tf.placeholder(tf.float32, [None, x_train.shape[1]])
Y = tf.placeholder(tf.float32)

# Set model weights
W = tf.Variable(tf.ones([ x_train.shape[1]]) )
b = tf.Variable(tf.zeros([1]))

# Construct a linear model
pred = tf.add(tf.matmul(X , W), b)

# # Mean squared error
# cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Mape
cost = tf.reduce_mean(tf.abs(tf.truediv(Y - pred, Y)))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        s_index = np.random.choice(x_train.shape[0], 200)
        x_train_s = x_train[s_index]
        y_train_s = y_train[s_index]

        sess.run(optimizer, feed_dict={X: x_train_s, Y: y_train_s})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_train, Y:y_train})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'
