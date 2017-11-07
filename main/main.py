import classify.prelim
from trec_car.read_data import *


def run():
    print("Running IR-Ext...")
    classify.prelim.do_classification()
    for p in iter_paragraphs("/home/aifs2/doylew/Projects/irext/input/train.test200.cbor.paragraphs"):
        print(p)


if __name__ == "__main__":
    run()
