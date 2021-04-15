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
         if not match:
            print("Predicted: {}| Actual: {}".format(hyp_val, actual))
         if i % 10 == 0:
            print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'
               .format(i+1, nb_queries, hyp_val, actual, match))
      
      accuracy = matches / nb_queries
      print('{} matches out of {} examples'.format(matches, nb_queries))
      return accuracy
    

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

data = pd.read_csv("data/iris.csv")
spec = data.groupby('species')

train = []
trainClasses = []
test = []
testClasses = []

for (k, group) in spec:
    l = int(len(group) * 0.8)
    i = [(x[0],x[1],x[2],x[3]) for x in group.to_numpy()]
    c = [classes.index(x[4]) for x in group.to_numpy()]
    train.extend(i[:l])
    trainClasses.extend(c[:l])
    test.extend(i[l:])
    testClasses.extend(c[l:])

nb_features = len(features)
nb_classes = len(classes)
k = 3

knn = KNN(nb_features, nb_classes, train, trainClasses, k, weighted = False)

accuracy =  knn.predict(test,testClasses)
print('Test set accuracy: ', accuracy)