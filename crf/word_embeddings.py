import os
import pickle

from gensim.models import KeyedVectors

from data import get_data

_CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

EN_MODEL = KeyedVectors.load_word2vec_format(
    os.path.join(_CURRENT_DIR_PATH, 'wiki.en', 'wiki.en.vec')
)


def get_fast_text_word_embedding(word):
    try:
        return EN_MODEL[word]
    except KeyError:
        return None


def get_clustering_model():
    with open(os.path.join(_CURRENT_DIR_PATH, 'wiki_fast_text_kmeans_500.pkl'), 'rb') as f_in:
        model = pickle.load(f_in, encoding='latin1')

    return model


def get_word_to_cluster_id_mapping():
    with open(os.path.join(_CURRENT_DIR_PATH, 'cluster_id_by_word.pkl'), 'rb') as _in:
        return pickle.load(_in)


if __name__ == '__main__':
    """
    The below code creates a dict of the form: <word>: <cluster_id>
    """
    train_data, validation_data, test_data = get_data()

    unique_words = set(
        [item[0] for _list in train_data + validation_data + test_data for item in _list]
    )

    embeddings_by_word = {word: get_fast_text_word_embedding(word) for word in unique_words}

    clustering_model = get_clustering_model()

    max_label = max(clustering_model.labels_)

    cluster_id_by_word = dict()
    for word, embedding in embeddings_by_word.items():
        if embedding is not None:
            cluster_id_by_word[word] = clustering_model.predict([embedding])[0]
        else:
            cluster_id_by_word[word] = max_label + 1

    with open('cluster_id_by_word.pkl', 'wb') as _out:
        pickle.dump(cluster_id_by_word, _out)
