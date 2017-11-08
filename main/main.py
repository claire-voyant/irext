import classify.prelim
import sys
from trec_car.read_data import *


def run():
    print("Running IR-Ext...")
    if len(sys.argv) < 1:
        print("usage ", sys.argv[0], " paragraphFile")
    classify.prelim.do_classification()
    with open(sys.argv[1], 'rb') as f:
        for p in iter_paragraphs(f):
            print('\n', p.para_id, ':')
            texts = [elem.text if isinstance(elem, ParaText)
                     else elem.anchor_text
                     for elem in p.bodies]
            print(' '.join(texts))


if __name__ == "__main__":
    run()
