from src.processing.InputParser import InputParser
from src.processing.PreProcessor import PreProcessor
from src.modelling.LSTMModel import LSTMModel


def train_model(training_data: str):
    parser = InputParser(training_data)
    sentences, ner_tags = parser.parse_data()

    pre_processor = PreProcessor(sentences, ner_tags)
    pre_processor.pre_process_data()

    lstm_model = LSTMModel(
        pre_processor.input_sequences,
        pre_processor.label_sequences,
        pre_processor.num_unique_word_tokens,
        pre_processor.num_unique_label_tokens,
        pre_processor.max_sequence_length
    )

    lstm_model.train()

    return lstm_model


if __name__ == "__main__":
    DATA = "../../conll2003/eng.train"
    model = train_model(DATA)