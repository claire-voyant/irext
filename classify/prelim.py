# Preliminary testing
import cbor
from IPython.display import display


def read_bson_file(f):
    b = cbor.load(open(f))
    for key in b.keys():
        print(key)


def do_classification():
    print("Doing some classification...")
