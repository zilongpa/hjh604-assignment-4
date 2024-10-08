from flask import Flask, render_template, request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# svd = TruncatedSVD(n_components=128)
svd = TruncatedSVD(n_components=2048)
X_reduced = svd.fit_transform(X)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        content = request.form.get('content')
        target_vec = vectorizer.transform([content])
        target_vec_reduced = svd.transform(target_vec)
        similarities = cosine_similarity(target_vec_reduced, X_reduced)
        top_5 = np.argsort(similarities[0])[-5:]

        return [{
            "id": int(i),
            "content": documents[i],
            "cosine_similarity": similarities[0][i]
        } for i in top_5]
