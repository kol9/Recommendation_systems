import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop


class BaseModel:

    def fit(self, positive_pairs, learning_rate, num_negatives):
        pass

    def predict(self, pairs, batch_size, verbose):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass


class MFModel(BaseModel):
    def __init__(self, num_user, num_item, embedding_dim, reg, stddev):
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = embedding_dim
        self.user_embedding = np.random.normal(0, stddev, (num_user, embedding_dim))
        self.item_embedding = np.random.normal(0, stddev, (num_item, embedding_dim))
        self.user_bias = np.zeros([num_user])
        self.item_bias = np.zeros([num_item])
        self.bias = 0.0
        self.reg = reg

    def _predict_one(self, user, item):
        return (self.bias + self.user_bias[user] + self.item_bias[item] +
                np.dot(self.user_embedding[user], self.item_embedding[item]))

    def predict(self, pairs, batch_size, verbose):
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        for i in range(num_examples):
            predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
        return predictions

    def fit(self, positive_pairs, learning_rate, num_negatives):
        user_item_label_matrix = self._convert_ratings_to_implicit_data(
            positive_pairs, num_negatives)
        np.random.shuffle(user_item_label_matrix)

        num_examples = user_item_label_matrix.shape[0]
        reg = self.reg
        lr = learning_rate
        sum_of_loss = 0.0
        for i in range(num_examples):
            (user, item, rating) = user_item_label_matrix[i, :]
            user_emb = self.user_embedding[user]
            item_emb = self.item_embedding[item]
            prediction = self._predict_one(user, item)

            if prediction > 0:
                one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
                sigmoid = 1.0 / one_plus_exp_minus_pred
                this_loss = (np.log(one_plus_exp_minus_pred) +
                             (1.0 - rating) * prediction)
            else:
                exp_pred = np.exp(prediction)
                sigmoid = exp_pred / (1.0 + exp_pred)
                this_loss = -rating * prediction + np.log(1.0 + exp_pred)

            grad = rating - sigmoid

            self.user_embedding[user, :] += lr * (grad * item_emb - reg * user_emb)
            self.item_embedding[item, :] += lr * (grad * user_emb - reg * item_emb)
            self.user_bias[user] += lr * (grad - reg * self.user_bias[user])
            self.item_bias[item] += lr * (grad - reg * self.item_bias[item])
            self.bias += lr * (grad - reg * self.bias)

            sum_of_loss += this_loss

        return sum_of_loss / num_examples

    def _convert_ratings_to_implicit_data(self, positive_pairs, num_negatives):
        num_items = self.item_embedding.shape[0]
        num_pos_examples = positive_pairs.shape[0]
        training_matrix = np.empty([num_pos_examples * (1 + num_negatives), 3],
                                   dtype=np.int32)
        index = 0
        for pos_index in range(num_pos_examples):
            u = positive_pairs[pos_index, 0]
            i = positive_pairs[pos_index, 1]

            training_matrix[index] = [u, i, 1]
            index += 1
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                training_matrix[index] = [u, j, 0]
                index += 1
        return training_matrix

    def _file_prefix(self):
        return 'trained_models/' + 'MFModel' + '_u' + str(self.num_user) + \
               '_i' + str(self.num_item) + \
               '_dim' + str(self.embedding_dim) + '_'

    def save_weights(self):
        file_prefix = self._file_prefix()
        np.save(file_prefix + 'user_embedding.npy', self.user_embedding)
        np.save(file_prefix + 'item_embedding.npy', self.item_embedding)
        np.save(file_prefix + 'user_bias.npy', self.user_bias)
        np.save(file_prefix + 'item_bias.npy', self.item_bias)
        np.save(file_prefix + 'bias.npy', [self.bias])

    def load_weights(self):
        file_prefix = self._file_prefix()
        self.user_embedding = np.load(file_prefix + 'user_embedding.npy')
        self.item_embedding = np.load(file_prefix + 'item_embedding.npy')
        self.user_bias = np.load(file_prefix + 'user_bias.npy')
        self.item_bias = np.load(file_prefix + 'item_bias.npy')
        self.bias = np.load(file_prefix + 'bias.npy')[0]


class MLPModel(BaseModel):
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8], reg_layers=[64, 32, 16, 8], learning_rate=0.01):
        self.layers = layers
        self.num_users = num_users
        self.num_items = num_items
        self.model = self._get_model(num_users, num_items, layers=layers, reg_layers=reg_layers)
        self.model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

    def _init_normal(self, shape):
        initializers.RandomNormal()
        return initializers.RandomNormal(shape, stddev=0.01)

    def _get_model(self, num_users, num_items, layers, reg_layers):
        # FIXME MODEL VERY SHIT, EBANIY V ROT!!!!!!!!!!!!
        assert len(layers) == len(reg_layers)
        self.num_layer = len(layers)
        num_layer = len(layers)

        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name='user_embedding',
                                       embeddings_initializer=self._init_normal,
                                       embeddings_regularizer=l2(reg_layers[0]), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='item_embedding',
                                       embeddings_initializer=self._init_normal,
                                       embeddings_regularizer=l2(reg_layers[0]), input_length=1)

        # Crucial to flatten an embedding vector!
        user_latent = Sequential([MLP_Embedding_User(user_input), Flatten()])
        item_latent = Sequential([MLP_Embedding_Item(user_input), Flatten()])

        # The 0-th layer is the concatenation of embedding layers
        vector = tf.keras.layers.Concatenate([user_latent, item_latent])

        # MLP layers

        for idx in range(1, num_layer):
            layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
            vector = layer(vector)

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

        model = Model(input=[user_input, item_input],
                      output=prediction)

        return model

    def _get_train_instances(self, train, num_negatives):
        user_input, item_input, labels = [], [], []
        num_users = train.shape[0]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(self.num_items)
                while train.has_key((u, j)):
                    j = np.random.randint(self.num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels

    def fit(self, train, learning_rate, num_negatives):
        user_input, item_input, labels = self._get_train_instances(train, num_negatives)
        self.model.fit([np.array(user_input), np.array(item_input)],  # input
                       np.array(labels),  # labels
                       batch_size=256, nb_epoch=1, verbose=0, shuffle=True)

    def predict(self, pairs, batch_size, verbose):
        return self.model.predict(pairs, batch_size=batch_size, verbose=verbose)

    def _file_name(self):
        return 'trained_models/' + 'MLPModel' + '_u' + str(self.num_users) + \
               '_i' + str(self.num_items) + \
               '_dim' + str(self.layers[0] / 2) + '_layers' + str(self.layers) + '_.h5'

    def save_weights(self):
        self.model.save_weights(self._file_name(), overwrite=True)

    def load_weights(self):
        pass
