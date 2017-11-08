# Preliminary testing
import cbor


def read_bson_file(f):
    b = cbor.load(open(f))
    for key in b.keys():
        print(key)


def do_classification():
    print("Doing some classification...")
