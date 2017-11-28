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
        for p in itertools.islice(iter_annotations(f), 500):
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

def dict_to_df(data):
    df = pd.DataFrame(list(data.items()))
    #print(df)
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

