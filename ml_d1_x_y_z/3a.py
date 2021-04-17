import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd

tf.disable_v2_behavior()


class KNN:
  
  def __init__(self, nb_features, nb_classes, trainData, trainDataClasses, k, weighted = False):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.trainData = trainData
    self.trainDataClasses = trainDataClasses
    self.k = k
    self.weight = weighted
    
    # Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
    self.TrainData = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.Classes = tf.placeholder(shape=(None), dtype=tf.int32)
    self.QueryVector = tf.placeholder(shape=(nb_features), dtype=tf.float32)
    
    # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.TrainData, self.QueryVector)), 
                                  axis=1))
    
    _, idxs = tf.nn.top_k(-dists, self.k)  
    
    self.classes = tf.gather(self.Classes, idxs)
    self.dists = tf.gather(dists, idxs)
    
    if weighted:
       self.w = 1 / self.dists  # Paziti na deljenje sa nulom.
    else:
       self.w = tf.fill([k], 1/k)
    
    # Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
    w_col = tf.reshape(self.w, (k, 1))
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)
    
    # Klasa sa najvise glasova je hipoteza.
    self.hyp = tf.argmax(self.scores)
  
  # Ako imamo odgovore za upit racunamo i accuracy.
  def predict(self, query_data):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
     
      nb_queries = len(query_data)
      
      matches = 0
      result_classes = []
      for i in range(nb_queries):
         hyp_val = sess.run(self.hyp, feed_dict = {self.TrainData: self.trainData, 
                                                  self.Classes: self.trainDataClasses, 
                                                  self.QueryVector: query_data[i]})
         result_classes.append(hyp_val)
      return result_classes
    

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

print(query_data)
print(query_classes)

knn = KNN(nb_features, nb_classes, train_data, train_classes, k, weighted = False)

result_classes =  knn.predict(query_data)

class_coloros = ['r', 'g', 'b']
class_back_coloros = ['orange', 'lightgreen', 'lightblue']

  # Generisemo grid.
step_size = 0.01

train_data_x = [a_tuple[0] for a_tuple in train_data]
train_data_y = [a_tuple[1] for a_tuple in train_data]


x1, x2 = np.meshgrid(np.arange(min(train_data_x), max(train_data_x),step_size),
                     np.arange(min(train_data_y), max(train_data_y), step_size))
x_feed = np.vstack((x1.flatten(), x2.flatten())).T

# Racunamo vrednost hipoteze.

pred_val = np.array(knn.predict(x_feed))
pred_plot = pred_val.reshape([x1.shape[0], x1.shape[1]])
  

# = pred_val.reshape([x1.shape[0], x1.shape[1]])

# Crtamo contour plot.
from matplotlib.colors import LinearSegmentedColormap

classes_cmap = LinearSegmentedColormap.from_list('classes_cmap', class_back_coloros)
plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)
  


for c in range(len(classes)):
   #p = [q.index(v) if v in q else 99999 for v in vm]
   x = [query_data[i][0]  for i,v in enumerate(result_classes) if v == c]
   y = [query_data[i][1]  for i,v in enumerate(result_classes) if v == c]
   plt.scatter(x,y, c=class_coloros[c], edgecolors='k', label=classes[c])
# plt.title("Accuracy {0:.0%}".format(accuracy))

plt.legend()
plt.show()
#  # Crtamo contour plot.
#   from matplotlib.colors import LinearSegmentedColormap
#   classes_cmap = LinearSegmentedColormap.from_list('classes_cmap', 
#                                                    ['lightblue', 
#                                                     'lightgreen', 
#                                                     'lightyellow'])
#   plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)
  
#   # Crtamo sve podatke preko.
#   idxs_0 = data['y'] == 0.0
#   idxs_1 = data['y'] == 1.0
#   idxs_2 = data['y'] == 2.0
#   plt.scatter(data['x'][idxs_0, 0], data['x'][idxs_0, 1], c='b', 
#               edgecolors='k', label='Klasa 0')
#   plt.scatter(data['x'][idxs_1, 0], data['x'][idxs_1, 1], c='g', 
#               edgecolors='k', label='Klasa 1')
#   plt.scatter(data['x'][idxs_2, 0], data['x'][idxs_2, 1], c='y', 
#               edgecolors='k', label='Klasa 2')
#   plt.legend()
#   # Uporediti plot za 1 i 100 epoha.