# Utility functions for classification and data processing

from itertools import islice
import pandas as pd

def take(n, iterable):
    "Return first n items of an iterable given as a list"
    return list(islice(iterable, n))

def dict_to_df(data):
    df = pd.DataFrame(list(data.items()))
    return df

def tuple_list_to_df(data):
    df = pd.DataFrame(data)
    return df

def create_tuples_from_cats(top_k_cats, train, test):
    # return a list of tuples
    # for both test and training which contains
    # (page_name, category) entries for ONLY those categories
    # in top_k_cats list, lists SHOULD repeat page_name if it
    # has multiple top categories in its category list!
    t_train = list()
    t_test = list()
    for (page_name, category_list) in train.items():
        for category in category_list:
            if category in top_k_cats:
                t_train.append((page_name, category))
    for (page_name, category_list) in test.items():
        for category in category_list:
            if category in top_k_cats:
                t_test.append((page_name, category))
    return (t_train, t_test)


