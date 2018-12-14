# NER Travel Queries
This includes two different implementations to solve the Airline Travel Information System(ATIS) Named-Entity-Recognition (NER) challenge.

One implementation uses (linear chain) conditional random fields (CRF), and, more specifically, the [python-crfsuite](http://python-crfsuite.readthedocs.org/en/latest/) library as its basis. 

The other implementation uses a single-layer LSTM leveraging Keras and Tensorflow.

The LSTM approach resulted in better performance in terms of precision and recall.

Here is an example sentence and its labels from the dataset:

  Show (O) | flights (O) | from (O) | Boston (B-dept) | to (O) | New (B-arr) | York (I-arr) | today (B-depart_date.today_relative)


# ATIS Data

Download ATIS Dataset here! [split 0](https://s3-eu-west-1.amazonaws.com/atis/atis.fold0.pkl.gz) [split 1](https://s3-eu-west-1.amazonaws.com/atis/atis.fold1.pkl.gz) [split 2](https://s3-eu-west-1.amazonaws.com/atis/atis.fold2.pkl.gz) [split 3](https://s3-eu-west-1.amazonaws.com/atis/atis.fold3.pkl.gz) [split 4](https://s3-eu-west-1.amazonaws.com/atis/atis.fold4.pkl.gz)

# Installation

-  Python 3.5
-  virtualenv venv && source venv/bin/activate
-  pip install -r requirements.txt
-  Execute `python download_data.py`
-  Execute `python -m spacy download en`
-  Specify tensorflow as a backend of Keras in file `~/.keras/keras.json` 

# Code

## CRF

- The python file `crf.training.py` performs Random Grid search (optimizing F1 score) to find the best possible values for parameters `c1` and `c2` of CRF. It saves the model in a file named `best_crf_model.pkl`. If you run it again, it will overwrite the `best_crf_model.pkl` file. It takes ~7 hours in ` MacBook Pro i7 16GB.
- The python file `crf.evaluation.py` evaluates the latter model in terms of Precision, Recall and Sequence Accuracy Score.
 
## LSTM (Long short-term memory neural network) 

- The python file `lstm.training.py` trains/evaluates a single-layer LSTM network. The weights for each epoch are stored under `lstm/keras_checkpoints/`. It takes a few hours in CPU at most.

# Performance

## CRF

- `Weighted Precision Score = 0.958208628893`
- `Weighted Recall Score = 0.96260056534`
- `Sequence Accuracy Score = 0.7928331466965286`
- `crf_results.txt` has thorough details about the performance of CRF in this task

## LSTM

- `Weighted Precision Score = 0.972639507033`
- `Weighted Recall Score = 0.97347249402`
- `Sequence Accuracy Score = 0.8286674132138858`
- `lstm_results.txt` has thorough details about the performance of LSTM in this task


# Future Work

Try more complicated LSTM architectures and/or use pre-trained word embeddings. 
