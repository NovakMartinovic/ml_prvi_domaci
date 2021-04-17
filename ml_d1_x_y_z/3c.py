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
  def predict(self, query_data, actual_classes):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
     
      nb_queries = len(query_data)
      
      matches = 0
      for i in range(nb_queries):
         hyp_val = sess.run(self.hyp, feed_dict = {self.TrainData: self.trainData, 
                                                  self.Classes: self.trainDataClasses, 
                                                  self.QueryVector: query_data[i]})
         actual = actual_classes[i]
         match = (hyp_val == actual)
         if match:
            matches += 1
            
      accuracy = matches / nb_queries
      return accuracy
    

features = ["sepal_length", "sepal_width" , "petal_length", "petal_width"]
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

print(query_data)
print(query_classes)

for k in range(1,16):

    knn = KNN(nb_features, nb_classes, train_data, train_classes, k, weighted = False)

    accuracy =  knn.predict(query_data,query_classes)
    print('K:', k, 'Accuracy ', accuracy)