try:
    import cPickle as pickle
except:
    import pickle
import os

from spacy.en import English


_DIR = os.path.dirname(os.path.realpath(__file__))
_PARSER = English()


def _load_data(fold_num):
    f = 'atis.fold%s.pkl' % fold_num
    with open(os.path.join(_DIR, f), 'rb') as in_file:
        train_set, valid_set, test_set, dicts = pickle.load(in_file, encoding='latin1')

    return train_set, valid_set, test_set, dicts


def _create_instances(word_indexes, label_indexes, idx2word, idx2label):
    words = []
    labels = []
    for wx, la in zip(word_indexes, label_indexes):
        word = idx2word[wx]
        label = idx2label[la]
        words.append(word)
        labels.append(label)

    sentence = ' '.join(words)
    tokens = _PARSER(sentence)

    pos_tags = [token.pos_ for token in tokens] if len(tokens) == len(words) else len(words) * ['X']

    instances_tuples = []
    for word, label, pos in zip(words, labels, pos_tags):
        instances_tuples.append((word, pos, label))

    return instances_tuples


def get_data_for_fold(fold_num):
    train_set, valid_set, test_set, dicts = _load_data(fold_num)

    w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

    idx2word = dict((v, k) for k, v in w2idx.items())
    idx2label = dict((v, k) for k, v in labels2idx.items())

    test_x, _, test_label = test_set
    validation_x, _, validation_label = valid_set
    train_x, _, train_label = train_set

    train_data, validation_data, test_data = None, None, None
    for e in ['train', 'validation', 'test']:
        all_instances = []
        for sw, sl in zip(eval(e + '_x'), eval(e + '_label')):
            instances_tuples = _create_instances(sw, sl, idx2word, idx2label)
            all_instances.append(instances_tuples)

        if e == 'train':
            train_data = all_instances
        elif e == 'validation':
            validation_data = all_instances
        else:
            test_data = all_instances

    return train_data, validation_data, test_data


def get_data():
    train_data, validation_data, test_data = [], [], []
    for i in range(5):
        fold_train_data, fold_validation_data, fold_test_data = get_data_for_fold(i)
        train_data.extend(fold_train_data)
        validation_data.extend(fold_validation_data)
        test_data.extend(fold_test_data)

    return train_data, validation_data, test_data
