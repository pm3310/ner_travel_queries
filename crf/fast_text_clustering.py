"""
This file clusters the Fast Text embeddings into 500 clusters.
It consumes a lot of memory. I ran it in a 40 core 128GB machine to cluster the embeddings.
"""
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from crf.word_embeddings import EN_MODEL

if __name__ == '__main__':
    word_embeddings = []
    for word in EN_MODEL.vocab:
        word_embeddings.append(EN_MODEL[word])

    kmeans = KMeans(n_clusters=500, n_jobs=-1).fit(word_embeddings)

    joblib.dump(kmeans, 'wiki_fast_text_kmeans_500.pkl')
