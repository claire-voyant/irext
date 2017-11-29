# Functions to read the data from the cbor file 
from trec_car.read_data import *
from itertools import islice

def process_data(file_location, data_samples = 20000):
    count = 0
    points = 0
    cat_keys = dict()
    keys_cats = dict()
    train = dict()
    test = dict()
    trainTest = 0
    # Open the unprocessed wiki text
    with open(file_location, 'rb') as f:
       # Loop over the first 100,000 wiki articles
        for p in itertools.islice(iter_annotations(f), data_samples):
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


