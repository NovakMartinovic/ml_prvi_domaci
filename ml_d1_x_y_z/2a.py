import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tf.disable_v2_behavior()


def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features+1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


filename = 'data/full_data.csv'

all_data = pd.read_csv(filename)
old = all_data[all_data['date'] <= '2020-10-21']
serbia = old[old['location'] == 'Serbia']
np_data = serbia.to_numpy()

nb_samples = np_data.shape[0]

data = dict()
data['x'] = np.arange(nb_samples)
data['y'] = np_data[:, 2]

indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / \
    np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

colors = ' bgrcmyk'
original_data = data.copy()

loss_result = np.zeros(6)

for nb_features in range(1, 7):
    data['x'] = create_feature_matrix(original_data['x'], nb_features)

    plt.scatter(data['x'][:, 0], data['y'])
    plt.xlabel('Date')
    plt.ylabel('Number of newly infected')

    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(nb_features))
    bias = tf.Variable(0.0)

    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)

    Y_col = tf.reshape(Y, (-1, 1))

    mse = tf.reduce_mean(tf.square(hyp - Y_col))

    opt_op = tf.train.AdamOptimizer().minimize(mse)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Izvršavamo 100 epoha treninga.
        nb_epochs = 100
        for epoch in range(nb_epochs):

            # Stochastic Gradient Descent.
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: data['x'][sample].reshape((1, nb_features)),
                        Y: data['y'][sample]}
                _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
                epoch_loss += curr_loss

            # U svakoj desetoj epohi ispisujemo prosečan loss.
            epoch_loss /= nb_samples
            if (epoch + 1) % 10 == 0:
                print('Epoch: {}/{}| Avg loss: {:.5f}'.format(epoch+1, nb_epochs,
                                                            epoch_loss))

        # Ispisujemo i plotujemo finalnu vrednost parametara.
        w_val = sess.run(w)
        bias_val = sess.run(bias)
        
        print('w = ', w_val, 'bias = ', bias_val)
        xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
        hyp_val = sess.run(hyp, feed_dict={X: xs})  # Bez Y jer nije potrebno.

        epoch_loss = 0
        for sample in range(nb_samples):
            feed = {X: data['x'][sample].reshape((1, nb_features)),
                    Y: data['y'][sample]}
            _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
            epoch_loss += curr_loss

        loss_result[nb_features-1] = epoch_loss 
        
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=colors[nb_features])
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
plt.show()

plt.plot(loss_result.T)
plt.xlabel('Exponent')
plt.ylabel('Loss function')
plt.show()

''' 
Funkcija troska pada do stepena 4, tu dobija svoj konacni oblik i daljim
povecanjem stepena se ne dobija primetna razlika.
'''