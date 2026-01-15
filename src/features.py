# src/features.py

from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(train_text, val_text, test_text,
                max_features=30000,
                ngram_range=(1,2)):
    """
    Fits TF-IDF on training data and transforms val & test data.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    return X_train, X_val, X_test, vectorizer
