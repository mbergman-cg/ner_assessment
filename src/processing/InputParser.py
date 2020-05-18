import os
from typing import Tuple


class InputParser:

    def __init__(self, input_data_path):
        self._input_data_path = input_data_path
        self._validate_file_path()

    def _validate_file_path(self):
        if not os.path.exists(self._input_data_path):
            raise FileNotFoundError(self._input_data_path)

    def parse_data(self) -> Tuple[list, list]:
        unique_labels = []

        sentences = []
        ner_tags = []

        sentence = []
        tag_sequence = []

        for line in open(self._input_data_path, 'r'):
            line = line.strip()
            if not line.startswith('-DOCSTART-'):
                if line != '':
                    parts = line.split(' ')
                    sentence.append(parts[0])
                    tag_sequence.append(parts[3])
                    if parts[3] not in unique_labels:
                        unique_labels.append(parts[3])
                else:
                    if len(sentence) != 0:
                        sentences.append(sentence)
                        ner_tags.append(tag_sequence)
                    sentence = []
                    tag_sequence = []

        return sentences, ner_tags
