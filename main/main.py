from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from trec_car.read_data import *
from itertools import islice
import sys
import itertools
import re
import pandas as pd
import numpy as np
import operator

def process_data():
    count = 0
    points = 0
    cat_keys = dict()
    keys_cats = dict()
    train = dict()
    test = dict()
    trainTest = 0
    if len(sys.argv) < 2:
        print("usage ", sys.argv[0], " articleFile")
    # Open the unprocessed wiki text
    with open(sys.argv[1], 'rb') as f:
       # Loop over the first 100,000 wiki articles
        for p in itertools.islice(iter_annotations(f), 20000):
            categorySet = set()
            # Ensure the wiki article has sections on the page
            if len(p.flat_headings_list()) > 0:
                # Go to the last section of the page
                section = p.flat_headings_list()[-1]
                # Ensure the last section has at least on paragraph
                if len(section[0].children) > 0:
                    categorySection = section[0].children[-1]
                    # Ensure the type of child is a paragraph, not a subsection
                    if type(categorySection) == Para:
                        # Split the paragraph by ') ', essentially splitting the category links for this page
                        if ') ' in str(categorySection):
                            links = str(categorySection).split(") ")
                            # Loop over each category link
                            for link in links:
                                # Split the category link and name
                                if (len(link.split('](')) > 1):
                                    link = link.split('](')[1]
                                    # Cleaning of category name due to splitting
                                    if ')' in link and '(' not in link:
                                        link = link.replace(')', '')
                                    if '))' in link:
                                        link = link.replace('))', ')')
                                    if '(' in link and ')' not in link:
                                        link = link + ')'
                                    # If this is a category, grab it
                                    if 'Category:' in link:
                                        category = link.replace('Category:', '')
                                        #print(p.page_name + ": " + category)
                                        #print('\n')
                                        if category not in cat_keys:
                                            count = count + 1
                                            cat_keys[category] = count
                                            keys_cats[count] = category
                                        # Add the category to the set for this page
                                        categorySet.add(cat_keys[category])
            if len(categorySet) > 0:
                points += len(categorySet)
                if trainTest % 2 == 0:
                    train[p.page_name] = categorySet
                else:
                    test[p.page_name] = categorySet
                trainTest += 1
                                        

    print(points, "data points")
    return (train, test, keys_cats)

def take(n, iterable):
    "Return first n items of an iterable given as a list"
    return list(islice(iterable, n))

def dict_to_df(data):
    df = pd.DataFrame(list(data.items()))
    #print(df)
    return df


def evaluate_accuracy(test_df, predicted, cat_map, format_string = ""):
    correct = 0.0
    number = 0.0
    for doc,category in zip(test_df.ix[:,1], predicted):
        if cat_map[doc] == cat_map[category]:
            correct = correct + 1.0
        number = number + 1.0
    print(format_string, " Accuracy:",float(correct/number))

def learn_top_k_categories(k = 10):
    (train, test, cat_map) = process_data()
    # keep a dictionary of counts for the category
    # 10,000 is an estimate of how many categories 
    category_counts = dict()
    for i in range(0,100000):
        # initialize them all to begin at count zero
        category_counts[i] = 0
    # iterate over the training set and count the categories
    for (page_name, categories) in train.items():
        for category in categories:
            category_counts[category] = category_counts[category] + 1
    # iterate over the test set and count the categories
    for (page_name, categories) in test.items():
        for category in categories: 
            category_counts[category] = category_counts[category] + 1
    # create a set for the top k categories
    top_k = set()
    # go through the category counts and find those which have
    # more than 20 data points, estimate can be changed
    for (category, count) in category_counts.items():
        if count > 20:
            if len(top_k) >= k:
                top_k_copy = top_k.copy()
                for cat in top_k_copy:
                    if category_counts[cat] < count:
                        top_k.remove(cat)
                        top_k.add(category)
            else:
                top_k.add(category)
    print("Top " + str(len(top_k)) + " categories retrieved!")
    return top_k

def create_tuples_from_cats(top_k_cats, train, test):
    # TODO implement this should return a list of
    # tuples for both test and training which contains
    # (page_name, category) entries for ONLY those categories
    # in top_k_cats list, lists SHOULD repeat page_name if it
    # has multiple top categories in its category list!
    return ([], [])


if __name__ == "__main__":
    print("Running IR-Ext...")

    # various parameters to be searched over for optimization
    parameters = {'vect__ngram_range': [(1,1), (1,2)],
                'tfidf__use_idf': (True, False),
                'clf__alpha': (1e-2, 1e-3),}

    # Naive Bayes pipeline for evaluation
    text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()),])

    # SVM (gradient descent) pipeline for evaluation
    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                            alpha=1e-3, max_iter=1000, tol=1e-3, random_state=42)),])

    # retrieve the top k categories
    # from the test and training data
    top_k_cats = learn_top_k_categories()

    # TODO create a dataframe with only the top categories
    (train, test, cat_map) = process_data()

    # create list of tuples using only top k categories
    (t_train, t_test) = create_tuples_from_cats(top_k_cats, train, test)

    train_df = dict_to_df(t_train)
    test_df = dict_to_df(t_test)

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

    # count matrix MBayes
    clf = MultinomialNB().fit(X_train_counts, train_df.ix[:,1])

    # tfidf matrix MBayes
    iclf = MultinomialNB().fit(X_train_tfidf, train_df.ix[:,1])
    text_clf = text_clf.fit(train_df.ix[:,0], train_df.ix[:,1])
    text_clf_svm.fit(train_df.ix[:,0], train_df.ix[:,1])
   
    X_new_counts = count_vect.transform(test_df.ix[:,0])

    predicted = clf.predict(X_new_counts)
    ipredicted = iclf.predict(X_new_counts)

    evaluate_accuracy(test_df, predicted, cat_map, format_string = "Count vector")
    evaluate_accuracy(test_df, ipredicted, cat_map, format_string = "TFIDF vector")

    predicted = text_clf.predict(test_df.ix[:,0])
    evaluate_accuracy(test_df, predicted, cat_map, format_string = "Pipeline vector")

    predicted = text_clf_svm.predict(test_df.ix[:,0])
    evaluate_accuracy(test_df, predicted, cat_map, format_string = "SVM TFIDF vector")
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=2)
    gs_clf = gs_clf.fit(train_df.ix[:,0], train_df.ix[:,1])

