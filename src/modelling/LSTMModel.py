import numpy as np

from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Sequential


class LSTMModel:

    def __init__(self,
                 train_input_sequences: np.ndarray,
                 train_label_sequences: np.ndarray,
                 num_unique_word_tokens: int,
                 num_unique_label_tokens: int,
                 max_sequence_length: int):
        self._train_input_sequences = train_input_sequences
        self._train_label_sequences = train_label_sequences
        self._model = None

        self._compile_model(num_unique_word_tokens, num_unique_label_tokens, max_sequence_length)

    def _compile_model(self,
                       num_unique_word_tokens: int,
                       num_unique_label_tokens: int,
                       max_sequence_length: int):
        embedding_size = 100

        self._model = Sequential([
            Embedding(num_unique_word_tokens, embedding_size, input_length=max_sequence_length,
                      mask_zero=True),
            Bidirectional(LSTM(50, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)),
            TimeDistributed(Dense(num_unique_label_tokens + 1, activation='softmax'))
        ])

        self._model.compile(loss='sparse_categorical_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])

    def train(self):
        history = self._model.fit(self._train_input_sequences, self._train_label_sequences,
                                  batch_size=50, epochs=1, verbose=1)
        print(history.history)
