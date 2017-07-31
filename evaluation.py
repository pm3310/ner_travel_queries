from sklearn_crfsuite.metrics import _flattens_y


@_flattens_y
def flat_recall_score(y_true, y_pred, **kwargs):
    """
    Return precision score for sequence items.
    """
    from sklearn import metrics
    return metrics.recall_score(y_true, y_pred, **kwargs)
