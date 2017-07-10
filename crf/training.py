import os
try:
    import cPickle as pickle
except:
    import pickle
import string
from collections import Counter

import numpy
import scipy
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from crf.airports import get_airport_codes, get_airport_names
from crf.countries import get_country_codes, get_country_names
from crf.regions import get_region_codes, get_region_names
from crf.word_embeddings import get_word_to_cluster_id_mapping
from data import get_data_for_fold


_CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

_WORD_TO_CLUSTER_ID_MAPPING = get_word_to_cluster_id_mapping()

_COUNTRY_CODES = get_country_codes()
_COUNTRY_NAMES_LIST = get_country_names()

_REGION_CODES = get_region_codes()
_REGION_NAMES_LIST = get_region_names()


_AIRPORT_CODES = get_airport_codes()
_AIRPORT_NAMES_LIST = get_airport_names()


def _is_country_code(input_text):
    return input_text.upper() in _COUNTRY_CODES


def _is_country_name_1(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[0]


def _is_country_name_2(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[1]


def _is_country_name_3(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[2]


def _is_country_name_4(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[3]


def _is_country_name_5(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[4]


def _is_country_name_6(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[5]


def _is_country_name_7(input_text):
    return input_text.lower() in _COUNTRY_NAMES_LIST[6]


def _is_region_code(input_text):
    return input_text.upper() in _REGION_CODES


def _is_region_name_1(input_text):
    return input_text.lower() in _REGION_NAMES_LIST[0]


def _is_region_name_2(input_text):
    return input_text.lower() in _REGION_NAMES_LIST[1]


def _is_region_name_3(input_text):
    return input_text.lower() in _REGION_NAMES_LIST[2]


def _is_region_name_4(input_text):
    return input_text.lower() in _REGION_NAMES_LIST[3]


def _is_region_name_5(input_text):
    return input_text.lower() in _REGION_NAMES_LIST[4]


def _is_region_name_6(input_text):
    return input_text.lower() in _REGION_NAMES_LIST[5]


def _is_airport_code(input_text):
    return input_text.upper() in _AIRPORT_CODES


def _is_airport_name_1(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[0]


def _is_airport_name_2(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[1]


def _is_airport_name_3(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[2]


def _is_airport_name_4(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[3]


def _is_airport_name_5(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[4]


def _is_airport_name_6(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[5]


def _is_airport_name_7(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[6]


def _is_airport_name_8(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[7]


def _is_airport_name_9(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[8]


def _is_airport_name_10(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[9]


def _is_airport_name_11(input_text):
    return input_text.lower() in _AIRPORT_NAMES_LIST[10]


def _looks_like_day_name(input_text):
    day_names = {
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'mondays', 'tuesdays', 'wednesdays', 'thursdays', 'fridays', 'saturdays', 'sundays',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    }

    return input_text.lower() in day_names


def _contains_only_digit_string(input_text):
    # That's because each digit is replaced with 'DIGIT' in the data set
    if len(input_text) % 5 != 0:
        return False

    for i in range(0, len(input_text), 5):
        if 'DIGIT' != input_text[i: i + 5]:
            return False

    return True


def _pm_or_am(input_text):
    return 'pm' == input_text.strip() or 'am' == input_text.strip()


def _contains_at_least_one_number(input_text):
    return any(char.isdigit() for char in input_text)


def _contains_at_least_one_punctuation(input_text):
    return any(char in string.punctuation for char in input_text)


def _contains_only_punctuation(input_text):
    return all(char in string.punctuation for char in input_text)


def _cluster_id(input_word):
    return _WORD_TO_CLUSTER_ID_MAPPING[input_word]


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'len(word)': len(word),
        '_contains_at_least_one_number(word)': _contains_at_least_one_number(word),
        '_contains_at_least_one_punctuation(word)': _contains_at_least_one_punctuation(word),
        '_contains_only_punctuation(word)': _contains_only_punctuation(word),
        '_contains_only_digit_string(word)': _contains_only_punctuation(word),
        '_pm_or_am(word)': _pm_or_am(word),
        '_looks_like_day_name(word)': _looks_like_day_name(word),
        '_is_country_code(word)': _is_country_code(word),
        '_is_country_name_1(word)': _is_country_name_1(word),
        '_is_country_name_2(word)': _is_country_name_2(word),
        '_is_country_name_3(word)': _is_country_name_3(word),
        '_is_country_name_4(word)': _is_country_name_4(word),
        '_is_country_name_5(word)': _is_country_name_5(word),
        '_is_country_name_6(word)': _is_country_name_6(word),
        '_is_country_name_7(word)': _is_country_name_7(word),
        '_is_region_code(word)': _is_region_code(word),
        '_is_region_name_1(word)': _is_region_name_1(word),
        '_is_region_name_2(word)': _is_region_name_2(word),
        '_is_region_name_3(word)': _is_region_name_3(word),
        '_is_region_name_4(word)': _is_region_name_4(word),
        '_is_region_name_5(word)': _is_region_name_5(word),
        '_is_region_name_6(word)': _is_region_name_6(word),
        '_is_airport_code(word)': _is_airport_code(word),
        '_is_airport_name_1(word)': _is_airport_name_1(word),
        '_is_airport_name_2(word)': _is_airport_name_2(word),
        '_is_airport_name_3(word)': _is_airport_name_3(word),
        '_is_airport_name_4(word)': _is_airport_name_4(word),
        '_is_airport_name_5(word)': _is_airport_name_5(word),
        '_is_airport_name_6(word)': _is_airport_name_6(word),
        '_is_airport_name_7(word)': _is_airport_name_7(word),
        '_is_airport_name_8(word)': _is_airport_name_8(word),
        '_is_airport_name_9(word)': _is_airport_name_9(word),
        '_is_airport_name_10(word)': _is_airport_name_10(word),
        '_is_airport_name_11(word)': _is_airport_name_11(word),
        '_cluster_id(word)': _cluster_id(word),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:len(word)': len(word1),
            '-1:_contains_at_least_one_number(word)': _contains_at_least_one_number(word1),
            '-1:_contains_at_least_one_punctuation(word)':
                _contains_at_least_one_punctuation(word1),
            '-1:_contains_only_punctuation(word)': _contains_only_punctuation(word1),
            '-1:_contains_only_digit_string(word)': _contains_only_punctuation(word1),
            '-1:_pm_or_am(word)': _pm_or_am(word1),
            '-1:looks_like_day_name(word)': _looks_like_day_name(word1),
            '-1:_is_country_code(word)': _is_country_code(word1),
            '-1:_is_country_name_1(word)': _is_country_name_1(word1),
            '-1:_is_country_name_2(word)': _is_country_name_2(word1),
            '-1:_is_country_name_3(word)': _is_country_name_3(word1),
            '-1:_is_country_name_4(word)': _is_country_name_4(word1),
            '-1:_is_country_name_5(word)': _is_country_name_5(word1),
            '-1:_is_country_name_6(word)': _is_country_name_6(word1),
            '-1:_is_country_name_7(word)': _is_country_name_7(word1),
            '-1:_is_region_code(word)': _is_region_code(word1),
            '-1:_is_region_name_1(word)': _is_region_name_1(word1),
            '-1:_is_region_name_2(word)': _is_region_name_2(word1),
            '-1:_is_region_name_3(word)': _is_region_name_3(word1),
            '-1:_is_region_name_4(word)': _is_region_name_4(word1),
            '-1:_is_region_name_5(word)': _is_region_name_5(word1),
            '-1:_is_region_name_6(word)': _is_region_name_6(word1),
            '-1:_is_airport_code(word)': _is_airport_code(word1),
            '-1:_is_airport_name_1(word)': _is_airport_name_1(word1),
            '-1:_is_airport_name_2(word)': _is_airport_name_2(word1),
            '-1:_is_airport_name_3(word)': _is_airport_name_3(word1),
            '-1:_is_airport_name_4(word)': _is_airport_name_4(word1),
            '-1:_is_airport_name_5(word)': _is_airport_name_5(word1),
            '-1:_is_airport_name_6(word)': _is_airport_name_6(word1),
            '-1:_is_airport_name_7(word)': _is_airport_name_7(word1),
            '-1:_is_airport_name_8(word)': _is_airport_name_8(word1),
            '-1:_is_airport_name_9(word)': _is_airport_name_9(word1),
            '-1:_is_airport_name_10(word)': _is_airport_name_10(word1),
            '-1:_is_airport_name_11(word)': _is_airport_name_11(word1),
            '-1:_cluster_id(word)': _cluster_id(word1),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:len(word)': len(word1),
            '+1:_contains_at_least_one_number(word)': _contains_at_least_one_number(word1),
            '+1:_contains_at_least_one_punctuation(word)':
                _contains_at_least_one_punctuation(word1),
            '+1:_contains_only_punctuation(word)': _contains_only_punctuation(word1),
            '+1:_contains_only_digit_string(word)': _contains_only_punctuation(word1),
            '+1:_pm_or_am(word)': _pm_or_am(word1),
            '+1:looks_like_day_name(word)': _looks_like_day_name(word1),
            '+1:_is_country_code(word)': _is_country_code(word1),
            '+1:_is_country_name_1(word)': _is_country_name_1(word1),
            '+1:_is_country_name_2(word)': _is_country_name_2(word1),
            '+1:_is_country_name_3(word)': _is_country_name_3(word1),
            '+1:_is_country_name_4(word)': _is_country_name_4(word1),
            '+1:_is_country_name_5(word)': _is_country_name_5(word1),
            '+1:_is_country_name_6(word)': _is_country_name_6(word1),
            '+1:_is_country_name_7(word)': _is_country_name_7(word1),
            '+1:_is_region_code(word)': _is_region_code(word1),
            '+1:_is_region_name_1(word)': _is_region_name_1(word1),
            '+1:_is_region_name_2(word)': _is_region_name_2(word1),
            '+1:_is_region_name_3(word)': _is_region_name_3(word1),
            '+1:_is_region_name_4(word)': _is_region_name_4(word1),
            '+1:_is_region_name_5(word)': _is_region_name_5(word1),
            '+1:_is_region_name_6(word)': _is_region_name_6(word1),
            '+1:_is_airport_code(word)': _is_airport_code(word1),
            '+1:_is_airport_name_1(word)': _is_airport_name_1(word1),
            '+1:_is_airport_name_2(word)': _is_airport_name_2(word1),
            '+1:_is_airport_name_3(word)': _is_airport_name_3(word1),
            '+1:_is_airport_name_4(word)': _is_airport_name_4(word1),
            '+1:_is_airport_name_5(word)': _is_airport_name_5(word1),
            '+1:_is_airport_name_6(word)': _is_airport_name_6(word1),
            '+1:_is_airport_name_7(word)': _is_airport_name_7(word1),
            '+1:_is_airport_name_8(word)': _is_airport_name_8(word1),
            '+1:_is_airport_name_9(word)': _is_airport_name_9(word1),
            '+1:_is_airport_name_10(word)': _is_airport_name_10(word1),
            '+1:_is_airport_name_11(word)': _is_airport_name_11(word1),
            '+1:_cluster_id(word)': _cluster_id(word1),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


if __name__ == '__main__':
    print("Started hyper-parameter optimisation!", end='\n\n')

    X_train, y_train, y_test = [], [], []
    cv_iterator = []
    max_num = 0

    # Prepare data for hyper-parameter optimisation
    for i in range(5):
        fold_train_data, fold_validation_data, fold_test_data = get_data_for_fold(i)

        X_fold_train_data = [sent2features(s) for s in fold_train_data]
        X_train.extend(X_fold_train_data)

        y_fold_train_data = [sent2labels(s) for s in fold_train_data]
        y_train.extend(y_fold_train_data)

        train_indices = numpy.arange(max_num, max_num + len(X_fold_train_data))
        max_num += len(X_fold_train_data)

        X_fold_validation_data = [sent2features(s) for s in fold_validation_data]
        X_train.extend(X_fold_validation_data)

        y_fold_validation_data = [sent2labels(s) for s in fold_validation_data]
        y_train.extend(y_fold_validation_data)

        validation_indices = numpy.arange(max_num, max_num + len(X_fold_validation_data))
        max_num += len(X_fold_validation_data)

        cv_iterator.append((train_indices, validation_indices))

        y_fold_test_data = [sent2labels(s) for s in fold_test_data]
        y_test.extend(y_fold_test_data)

    print("Prepared data!", end='\n\n')

    labels = list(set([label for labels in y_train + y_test for label in labels]))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=sorted_labels)

    rs = RandomizedSearchCV(crf, params_space,
                            cv=cv_iterator,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_, end='\n\n')
    print('best CV score:', rs.best_score_, end='\n\n')
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000), end='\n\n')

    crf = rs.best_estimator_

    with open("best_crf_model.pkl", "wb") as out_file:
        pickle.dump(crf, out_file)

    print("Trained model just persisted!", end='\n\n')

    print("Top likely transitions:", end='\n\n')
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:", end='\n\n')
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])
