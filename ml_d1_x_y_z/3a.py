import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

tf.disable_v2_behavior()

# knn = KNN(nb_features, nb_classes, train_data, train_classes, k, weighted = False)
class KNN:
  
  def __init__(self, nb_features, nb_classes, train_data, train_classes, k, weighted = False):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.train_data = train_data
    self.train_classes = train_classes
    self.k = k
    self.weight = weighted
    
    self.in_tain_data = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.in_train_classes = tf.placeholder(shape=(None), dtype=tf.int32)
    self.query_vector = tf.placeholder(shape=(nb_features), dtype=tf.float32)
    

    
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.in_tain_data, self.query_vector)), axis=1))
    
    _, idxs = tf.nn.top_k(-dists, self.k)  
    
    self.classes = tf.gather(self.in_train_classes, idxs)
    self.dists = tf.gather(dists, idxs)
    
    if weighted:
       self.w = 1 / self.dists  # Paziti na deljenje sa nulom.
    else:
       self.w = tf.fill([k], 1/k)
    
    w_col = tf.reshape(self.w, (k, 1))
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)
    
    self.predicted_class = tf.argmax(self.scores)
  

  def predict(self, query_data):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
     
      nb_queries = len(query_data)
      
      matches = 0
      result_classes = []
      for i in range(nb_queries):
         hyp_val = sess.run(self.predicted_class, 
            feed_dict = {
                           self.in_tain_data: self.train_data, 
                           self.in_train_classes: self.train_classes, 
                           self.query_vector: query_data[i]
                        })
         result_classes.append(hyp_val)
      return np.array(result_classes)
    

features = ["sepal_length", "sepal_width"] #, "petal_length", "petal_width"
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

data = pd.read_csv("data/iris.csv")
spec = data.groupby('species')

train_data = []
train_classes = []
query_data = []
query_classes = []

for (group_class, group) in spec:
   if(group_class not in classes):
      continue
   class_index = classes.index(group_class)
   l = int(len(group) * 0.8)
   i = []
   c = []
   for item in group.itertuples():
      t = []
      for feature in features:
         t.append(getattr(item, feature))
      i.append(tuple(t))
      c.append(class_index)

   train_data.extend(i[:l])
   train_classes.extend(c[:l])
   query_data.extend(i[l:])
   query_classes.extend(c[l:])

nb_features = len(features)
nb_classes = len(classes)
k = 3

knn = KNN(nb_features, nb_classes, train_data, train_classes, k, weighted = False)

result_classes =  knn.predict(query_data)

class_coloros = ['r', 'g', 'b']
class_back_coloros = ['orange', 'lightgreen', 'lightblue']

# step_size = 0.01 mnogo je sporo sa ovim accuracijem
step_size = 0.03

train_data_x = [t[0] for t in query_data]
train_data_y = [t[1] for t in query_data]

x1, x2 = np.meshgrid(np.arange(min(train_data_x), max(train_data_x), step_size),
                     np.arange(min(train_data_y), max(train_data_y), step_size))


x_feed = np.vstack((x1.flatten(), x2.flatten())).T

pred_val = np.array(knn.predict(x_feed))
pred_plot = pred_val.reshape([x1.shape[0], x1.shape[1]])

classes_cmap = LinearSegmentedColormap.from_list('classes_cmap', class_back_coloros)
#plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)

for c in range(len(classes)):
   x = [train_data_x[i] for i,v in enumerate(result_classes) if v == c]
   y = [train_data_y[i] for i,v in enumerate(result_classes) if v == c]
   plt.scatter(x,y, c=class_coloros[c], edgecolors='k', label=classes[c])


accuracy = np.mean(result_classes == query_classes)
plt.title("Accuracy {0:.0%}".format(accuracy))

plt.legend()
plt.show()