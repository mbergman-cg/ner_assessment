import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class PreProcessor:

    def __init__(self, sentences, ner_tags, oov_token: str = "<OOV>"):
        self._sentences = sentences
        self._ner_tags = ner_tags
        self._input_sequences = None
        self._label_sequences = None
        self._tokenizer = Tokenizer(oov_token=oov_token)
        self._label_tokenizer = Tokenizer()
        self._max_sequence_length = None

    def pre_process_data(self):
        self._pre_process_input_sequences()
        self._pre_process_label_sequences()

    def _pre_process_input_sequences(self):
        self._tokenizer.fit_on_texts(self._sentences)
        input_sequences = self._tokenizer.texts_to_sequences(self._sentences)
        padded_input_sequences = pad_sequences(input_sequences, padding='post')

        self._compute_max_sequence_len(input_sequences)
        self._input_sequences = np.array(padded_input_sequences)

    def _pre_process_label_sequences(self):
        self._label_tokenizer.fit_on_texts(self._ner_tags)
        label_sequences = self._label_tokenizer.texts_to_sequences(self._ner_tags)
        padded_label_sequences = pad_sequences(label_sequences, padding='post')

        self._label_sequences = np.array(padded_label_sequences)

    def _compute_max_sequence_len(self, input_sequences):
        seq_lengths = [len(seq) for seq in input_sequences]
        self._max_sequence_length = max(seq_lengths)

    @property
    def input_sequences(self):
        return self._input_sequences

    @property
    def label_sequences(self):
        return self._label_sequences

    @property
    def num_unique_word_tokens(self):
        return len(self._tokenizer.word_index)

    @property
    def num_unique_label_tokens(self):
        return len(self._label_tokenizer.word_index)
