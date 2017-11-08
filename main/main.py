import classify.prelim
import sys
from trec_car.read_data import *


def print_data():
    if len(sys.argv) < 4:
        print("usage ", sys.argv[0], " articleFile outlineFile paragraphFile")
    classify.prelim.do_classification()
    with open(sys.argv[1], 'rb') as f:
        for p in iter_annotations(f):
            print('\npagename: ', p.page_name)
            print('\npageid: ', p.page_id)
            print('\nmeta: ', p.page_meta)

    with open(sys.argv[2], 'rb') as f:
        for p in iter_annotations(f):
            print('\npagename:', p.page_name)
            for heading in p.outline():
                print(heading)

    with open(sys.argv[3], 'rb') as f:
        for p in iter_paragraphs(f):
            print('\n', p.para_id, ':')
            texts = [elem.text if isinstance(elem, ParaText)
                     else elem.anchor_text
                     for elem in p.bodies]
            print(' '.join(texts))


if __name__ == "__main__":
    print("Running IR-Ext...")
    print_data()
