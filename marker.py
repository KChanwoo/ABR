from network.RNN import Net
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

net = Net('./data/Response-result-origin.txt')

index = 0

data_vector = net.get_test_data()[0][index:index + 1]
data_label = net.get_test_data()[1][index:index + 1]

predict = net.predict(300, data_vector)

data_for_graph = []

for i in range(len(data_vector[0])):
    data_for_graph.append(data_vector[0][i][0])

label_val = int(np.argmax(data_label[0].numpy()))
predict_index = int(tf.math.argmax(predict, 1)[0])

plt.plot(range(len(data_for_graph)), data_for_graph, marker='', color='teal', linewidth=2)
plt.scatter(label_val, data_for_graph[label_val])
plt.scatter(predict_index, data_for_graph[predict_index], edgecolors='red')

print(label_val, predict_index)
plt.legend()
plt.show()
