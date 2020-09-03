import tensorflow as tf
import numpy as np
import network.model as m


class Net:
    def __init__(self, data_path):
        data_all = [[], [], []]
        with open(data_path) as f:
            lines = f.readlines()

            for i in range(len(lines)):
                line = lines[i]
                data_split = line.split(',')
                data_vector = []
                for j in range(1, len(data_split) - 2):
                    # model no.1
                    data_vector.append([int(data_split[j])])
                    # model no.2
                    # data_vector.append(int(data_split[j]))

                vector = data_vector
                answer = int(data_split[len(data_split) - 2][2:])

                if answer < len(vector):
                    # model no.1
                    data_all[0].append(data_vector)
                    max_value = max(data_vector)
                    new_list = []
                    for j in range(len(data_vector)):
                        new_list.append([data_vector[j][0] / max_value[0]])
                    data_all[2].append(new_list)

                    # data_all[1].append([answer, data_vector[answer][0]])

                    # model no.1 - 2
                    zeros = [0] * len(data_vector)
                    zeros[answer] = 1
                    data_all[1].append(zeros)

                    # model no.2
                    # image = np.zeros([495, 1000, 1])
                    # for k in range(len(data_vector)):
                    #     image[k][data_vector[k]][0] = 1
                    # data_all[0].append(image)
                    # data_all[1].append([answer, data_vector[answer]])

            sp = int(len(data_all[0]) * 0.8)
            self._train_data = [tf.convert_to_tensor(data_all[2][:sp], dtype=tf.float32), tf.convert_to_tensor(data_all[1][:sp], dtype=tf.float32)]
            self._test_data = [tf.convert_to_tensor(data_all[2][sp:], dtype=tf.float32), tf.convert_to_tensor(data_all[1][sp:], dtype=tf.float32)]

        self._time_step = self._train_data[0].shape[1]
        self._model = None

    def build_model(self, force=False):
        if self._model is None or force:
            self._model = m.model_test_1_2(self._time_step)
            # self._model = m.model_test_2(495, 1000)
            self._model.summary()

    def vector_to_data(self, vector_list):
        tensor_list = []
        for vector in vector_list:
            max_value = max(vector)
            change = [x / max_value for x in vector]

            tensor_list.append(change[: 495])

        tensor = tf.convert_to_tensor(tensor_list)
        tensor = tf.reshape(tensor, (len(tensor_list), len(tensor_list[0]), 1))

        return tensor

    def to_top_predict(self, prediction):
        top_list = []

        index_list = tf.math.argmax(prediction, 1).numpy()
        for i in range(len(index_list)):
            index = index_list[i]
            top_list.append({
                "prediction": int(index),
                "score": float(prediction[i, index].numpy())
            })

        return top_list

    def test(self, index):
        self.build_model()
        self._model.load_weights('./checkpoints/ABR_{}.tf'.format(index))

        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        res = self._model(self._test_data[0], training=False)

        predict_index = tf.math.argmax(res, 1)

        label_val = tf.math.argmax(self._test_data[1], 1)
        loss = tf.keras.losses.MSE(label_val, predict_index)
        # loss = tf.keras.losses.categorical_crossentropy(self._test_data[1], res)
        avg_loss.update_state(loss)

        avg_loss_value = avg_loss.result().numpy()

        avg_loss.reset_states()

        self._model.reset_states()

        return avg_loss_value, res

    def predict(self, index, data):
        self.build_model()

        self._model.load_weights('./checkpoints/ABR_{}.tf'.format(index))
        return self._model(data)

    def train(self, epoch=10000, batch_size=32, lr=0.001):
        self.build_model(True)

        optimizer = tf.keras.optimizers.Adam(lr=lr)

        iter = round(len(self._train_data[0]) / batch_size)

        for i in range(epoch):
            avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            avg_mse = tf.keras.metrics.Mean('mse', dtype=tf.float32)

            for j in range(iter):

                labels = self._train_data[1][j * batch_size: j * batch_size + batch_size]
                inputs = self._train_data[0][j * batch_size: j * batch_size + batch_size]

                with tf.GradientTape() as tape:
                    outputs = self._model(inputs, training=True)
                    # loss = tf.keras.losses.MSE(labels, outputs)  # calculate loss using MSE
                    # loss = tf.keras.losses.categorical_crossentropy(labels, outputs)
                    loss = tf.keras.losses.cosine_similarity(labels, outputs, axis=1)

                    predict_index = tf.math.argmax(outputs, 1)

                    label_val = tf.math.argmax(labels, 1)
                    mse = tf.keras.losses.MSE(label_val, predict_index)

                grads = tape.gradient(loss, self._model.trainable_variables)  # calculate gradients
                optimizer.apply_gradients(zip(grads, self._model.trainable_variables))  # update gradients

                avg_loss.update_state(loss)
                avg_mse.update_state(mse)

            avg_loss_value = avg_loss.result().numpy()
            avg_mse_value = avg_mse.result().numpy()
            print('Epoch: {} Cost: {} MSE: {}'.format(i, avg_loss_value, avg_mse_value))

            # save weight every 100 epochs
            if i % 100 == 0 and i != 0:
                self._model.save_weights('./checkpoints/ABR_{}.tf'.format(i))

            avg_loss.reset_states()
            avg_mse.reset_states()

    def get_test_data(self):
        return self._test_data

    def get_train_data(self):
        return self._train_data
