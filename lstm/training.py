from lstm.data_reader import DataReader
from lstm.ner_model import NERModelTrainer


if __name__ == '__main__':
    data_reader = DataReader()
    ner_model_trainer = NERModelTrainer(data_reader)
    ner_model_trainer.train(epochs=30, batch=30, dropout=0.15, lstm_units=200)
    ner_model_trainer.evaluate()
