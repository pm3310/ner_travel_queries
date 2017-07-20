import pickle

from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import _flattens_y

from crf.training import sent2features, sent2labels
from data import get_data


@_flattens_y
def flat_recall_score(y_true, y_pred, **kwargs):
    """
    Return precision score for sequence items.
    """
    from sklearn import metrics
    return metrics.recall_score(y_true, y_pred, **kwargs)


if __name__ == '__main__':
    print(
        "Evaluation started! It will take some time in order to load the Fast Text model.",
        end='\n\n'
    )
    train_data, validation_data, test_data = get_data()
    X_test = [sent2features(s) for s in test_data]
    y_test = [sent2labels(s) for s in test_data]

    y_train = [sent2labels(s) for s in train_data]
    y_validation = [sent2labels(s) for s in validation_data]

    labels = list(set([label for labels in y_train + y_test for label in labels]))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    with open("best_crf_model.pkl", "rb") as in_file:
        crf = pickle.load(in_file)

    y_pred = crf.predict(X_test)

    print("### Classification Report ###")
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels), end='\n\n')

    print("### Sequence Accuracy Score ###")
    print(metrics.sequence_accuracy_score(y_test, y_pred), end='\n\n')

    print("### Weighted Precision Score ###")
    print(metrics.flat_precision_score(y_test, y_pred, average='weighted'), end='\n\n')

    print("### Weighted Recall Score ###")
    print(flat_recall_score(y_test, y_pred, average='weighted'), end='\n\n')
