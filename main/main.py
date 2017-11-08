import classify.prelim
from trec_car.read_data import *


def run():
    print("Running IR-Ext...")
    classify.prelim.do_classification()
    with open("/home/aifs2/doylew/Projects/irext/input/train.test200.cbor.paragraphs", 'rb') as f:
        for p in iter_paragraphs(f):
            print('\n', p.para_id, ':')
            texts = [elem.text if isinstance(elem, ParaText)
                     else elem.anchor_text
                     for elem in p.bodies]
            print(' '.join(texts))


if __name__ == "__main__":
    run()
