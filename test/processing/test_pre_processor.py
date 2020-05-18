import numpy as np
from src.processing.PreProcessor import PreProcessor


def test_pre_process_data():
    in_sentences = [
        ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE',
         'DEFEAT', '.'],
        ['Nadim', 'Ladki']
    ]

    in_tags = [
        ['O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'I-PER', 'O', 'O', 'O', 'O'],
        ['I-PER', 'I-PER']
    ]

    pre_processor = PreProcessor(in_sentences, in_tags)
    pre_processor.pre_process_data()

    expected_input_sequences = np.array([
        [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
        [14, 15, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ])

    expected_label_sequences = np.array([
        [1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    expected_num_unique_word_tokens = 15  # Unique words + one oov token

    expected_num_unique_label_tokens = 3

    assert (pre_processor.input_sequences == expected_input_sequences).all()
    assert (pre_processor.label_sequences == expected_label_sequences).all()
    assert pre_processor.num_unique_word_tokens == expected_num_unique_word_tokens
    assert pre_processor.num_unique_label_tokens == expected_num_unique_label_tokens
