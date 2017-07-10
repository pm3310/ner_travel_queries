# NER Travel Queries
This is an implementation using (linear chain) conditional random fields (CRF) in python 3.5 for named entity recognition (NER). It uses the [python-crfsuite](http://python-crfsuite.readthedocs.org/en/latest/) library as its basis. 
It solves the Airline Travel Information System(ATIS) dataset.

Here is an example sentence and its labels from the dataset:

  Show (O) | flights (O) | from (O) | Boston (B-dept) | to (O) | New (B-arr) | York (I-arr) | today (O)


# ATIS Data

Download ATIS Dataset here! [split 0](https://s3-eu-west-1.amazonaws.com/atis/atis.fold0.pkl.gz) [split 1](https://s3-eu-west-1.amazonaws.com/atis/atis.fold1.pkl.gz) [split 2](https://s3-eu-west-1.amazonaws.com/atis/atis.fold2.pkl.gz) [split 3](https://s3-eu-west-1.amazonaws.com/atis/atis.fold3.pkl.gz) [split 4](https://s3-eu-west-1.amazonaws.com/atis/atis.fold4.pkl.gz)

# Installation

-  Python 3.5
-  virtualenv venv && source venv/bin/activate
-  pip install -r requirements.txt
-  Execute `python download_data.py`
-  Execute `python -m spacy download en`

# Code

- The python file `training.py` performs Random Grid search (optimizing F1 score) to find the best possible values for parameters `c1` and `c2` of CRF. It saves the model in a file named `best_crf_model.pkl`. If you run it again, it will overwrite the `best_crf_model.pkl` file. It takes ~7 hours in ` MacBook i7 16GB.
- The python file `evaluation.py` evaluates the latter model in terms of Precision, Recall and Sequence Accuracy Score.
 
# Performance

- `Weighted Precision Score = 0.958208628893`
- `Weighted Recall Score = 0.96260056534`
- `Sequence Accuracy Score = 0.7928331466965286`
- `results.txt` has thorough details about the performance of CRF in this task

# Future Work

Try some form of LSTM Network to avoid feature engineering!
