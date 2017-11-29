from read import process_data
from util import *
from classify import *
import sys
import re
import pandas as pd
import numpy as np
import operator
import argparse

if __name__ == "__main__":
    print("Running IR-Ext...")

    parser = argparse.ArgumentParser(description='Evaluating Category Classification in Information Retrieval')
    parser.add_argument('datafile', metavar='datafile', type=str, nargs='+',
            help='the unprocessed cbor data file for Wikipedia pages')
    parser.add_argument("--samples", help="number of samples to pull from the dataset")
    args = parser.parse_args()

    n_samples = 20000

    if args.samples is not None:
        n_samples = int(args.samples)

    print("Using %d samples from the data!" % n_samples)
    # initial read of the data to be processed
    (train, test, cat_map) = process_data(args.datafile[0], data_samples = n_samples)

    # retrieve the top k categories
    # from the test and training data
    top_k_cats = learn_top_k_categories(train, test, cat_map)

    # create list of tuples using only top k categories
    (t_train, t_test) = create_tuples_from_cats(top_k_cats, train, test)
    print(str(len(t_train)) + " training data points")
    # print(t_train)
    # print("************************************************************")
    print(str(len(t_test)) + " testing data points")
    # print(t_test)

    # create a dataframe with only the top categories
    train_df = tuple_list_to_df(t_train)
    test_df = tuple_list_to_df(t_test)

    # print(train_df.ix[:,0])

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(train_df.ix[:,0])
    # print(X_train_counts)
    # print("Count matrix...")
    # X_train_counts.shape
    # print(count_vect.vocabulary_.get(u'algorithm'))

    # tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    # X_train_tf = tf_transformer.transform(X_train_counts)
    # print("TF matrix...")
    # X_train_tf.shape

    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print("TFIDF matrix...")
    # X_train_tfidf.shape

    # count matrix MBayes
    # clf = MultinomialNB().fit(X_train_counts, train_df.ix[:,1])

    # tfidf matrix MBayes
    # iclf = MultinomialNB().fit(X_train_tfidf, train_df.ix[:,1])
   
    # X_new_counts = count_vect.transform(test_df.ix[:,0])

    # predicted = clf.predict(X_new_counts)
    # ipredicted = iclf.predict(X_new_counts)

    run_naive_bayes(train_df, test_df, cat_map)
    run_svm(train_df, test_df, cat_map)
    run_multi_svm(train_df, test_df, cat_map)
    run_multi_naive_bayes(train_df, test_df, cat_map)

    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=2)
    # gs_clf = gs_clf.fit(train_df.ix[:,0], train_df.ix[:,1])

