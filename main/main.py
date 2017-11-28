from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import sys
from trec_car.read_data import *
import itertools
import re
import pandas as pd

def process_data():
    count = 0
    cat_keys = dict()
    keys_cats = dict()
    train = dict()
    test = dict()
    if len(sys.argv) < 1:
        print("usage ", sys.argv[0], " articleFile")
    with open(sys.argv[1], 'rb') as f:
        data_slice = itertools.islice(iter_annotations(f), 2000)
        for p in data_slice:
            saved_article = ""
            for s in p.skeleton:
                n_s = str(s)
                if "Category" in n_s:
                    f_s = n_s[n_s.find("Category"):]
                    slice_f = f_s[:f_s.find("]")]
                    cat_slice = slice_f[len("Category:"):]
                    if "Category" not in cat_slice and "[" not in cat_slice:
                        count = count + 1
                        category = slice_f[len("Category:"):]
                        if p.page_name not in cat_keys:
                            cat_keys[category] = count
                            keys_cats[count] = category
                        if count % 10 == 0:
                            test[p.page_name] = cat_keys[category]
                        else:
                            train[p.page_name] = cat_keys[category]
    print(count, "data points")
    return (train, test, keys_cats)

def dict_to_df(data):
    df = pd.DataFrame(list(data.items()))
    return df


def evaluate_accuracy(test_df, predicted, cat_map):
    correct = 0.0
    number = 0.0
    for doc,category in zip(test_df.ix[:,1], predicted):
        if cat_map[doc] == cat_map[category]:
            correct = correct + 1.0
        number = number + 1.0
    print("Accuracy:",float(correct/number))


if __name__ == "__main__":
    print("Running IR-Ext...")

    (train, test, cat_map) = process_data()
    train_df = dict_to_df(train)
    test_df = dict_to_df(test)

    #print(train_df.ix[:,0])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_df.ix[:,0])
    #print(X_train_counts)
    print("Count matrix...")
    X_train_counts.shape
    #print(count_vect.vocabulary_.get(u'algorithm'))

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print("TF matrix...")
    X_train_tf.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print("TFIDF matrix...")
    X_train_tfidf.shape

    # count vector MBayes
    clf = MultinomialNB().fit(X_train_counts, train_df.ix[:,1])

    # tfidf matrix MBayes
    iclf = MultinomialNB().fit(X_train_tfidf, train_df.ix[:,1])

    text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()),])
    
    X_new_counts = count_vect.transform(test_df.ix[:,0])

    predicted = clf.predict(X_new_counts)
    ipredicted = iclf.predict(X_new_counts)

    evaluate_accuracy(test_df, predicted, cat_map)
    evaluate_accuracy(test_df, ipredicted, cat_map)

