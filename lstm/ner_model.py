import os
import shutil

import numpy
import pathlib
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM, TimeDistributed, Dense
from keras.models import Sequential, load_model
from sklearn_crfsuite import metrics

from evaluation import flat_recall_score

_CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class NERModelTrainer(object):
    def __init__(self, data_reader):
        self._data_reader = data_reader
        self._model = None

    def train(self, epochs=10, lstm_units=100, batch=20, dropout=0.0, word_embedding_size=200):
        self._model = Sequential()
        self._model.add(
            Embedding(
                self._data_reader.n_vocabulary,
                word_embedding_size,
                input_length=self._data_reader.max_train_sentence_length,
                mask_zero=True
            )
        )
        self._model.add(
            LSTM(
                units=lstm_units,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=True,
                input_shape=(self._data_reader.max_train_sentence_length, word_embedding_size)
            )
        )
        self._model.add(
            TimeDistributed(
                Dense(self._data_reader.n_classes, activation='softmax'),
                input_shape=(self._data_reader.max_train_sentence_length, word_embedding_size)
            )
        )
        self._model.compile(loss='categorical_crossentropy', optimizer='adam')

        print(self._model.summary())

        path_for_checkpoints = os.path.join(_CURRENT_DIR_PATH, 'keras_checkpoints')

        if os.path.exists(path_for_checkpoints):
            shutil.rmtree(path_for_checkpoints)

        pathlib.Path(path_for_checkpoints).mkdir()

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(
                path_for_checkpoints, 'checkpoint-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
            )
        )

        train_history = self._model.fit(
            x=self._data_reader.train_X,
            y=self._data_reader.train_y,
            epochs=epochs,
            batch_size=batch,
            validation_data=(self._data_reader.validation_X, self._data_reader.validation_y),
            callbacks=[checkpoint]
        )

        loss = train_history.history['loss']
        print("History Loss")
        print(loss)
        val_loss = train_history.history['val_loss']
        print("History Validation Loss")
        print(val_loss)

    def evaluate(self):
        loss = self._model.evaluate(self._data_reader.test_X, self._data_reader.test_y)
        print()
        print('Loss is: %f' % loss)

        all_predicted_labels = []
        all_true_labels = []
        for i, _test_instance in enumerate(self._data_reader.test_X):
            test_prediction = self._model.predict(_test_instance.reshape(
                1, self._data_reader.max_train_sentence_length)
            )[0]

            predicted_labels, true_labels = [], []
            for encoded_true_label_array, encoded_test_label_array in zip(
                    self._data_reader.test_y[i], test_prediction
            ):
                contains_all_zeros = not numpy.any(encoded_true_label_array)
                if not contains_all_zeros:
                    predicted_labels.append(
                        self._data_reader.decode_single_label(encoded_test_label_array)
                    )
                    true_labels.append(
                        self._data_reader.decode_single_label(encoded_true_label_array)
                    )

            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)

        print("### Classification Report ###")
        print(metrics.flat_classification_report(
            all_true_labels, all_predicted_labels, labels=self._data_reader.labels), end='\n\n'
        )

        print("### Sequence Accuracy Score ###")
        print(metrics.sequence_accuracy_score(all_true_labels, all_predicted_labels), end='\n\n')

        print("### Weighted Precision Score ###")
        print(metrics.flat_precision_score(
            all_true_labels, all_predicted_labels, average='weighted'), end='\n\n'
        )

        print("### Weighted Recall Score ###")
        print(flat_recall_score(
            all_true_labels, all_predicted_labels, average='weighted'), end='\n\n'
        )


class NERModelPredictor(object):
    def __init__(self, data_reader, path_to_model_file):
        self._data_reader = data_reader
        self._model = load_model(path_to_model_file)

    def predict(self, sentence_list):
        encoded_sentence = self._data_reader.encode_sentence(sentence_list)

        encoded_prediction = self._model.predict(encoded_sentence)

        return self._data_reader.decode_labels(encoded_prediction[0][-len(sentence_list):])
