# irext

Evaluation of Categorical Labelling of Wikipedia Pages

There is one package inside of ```classify``` which contains the ```main.py```
script which runs the program.

It requires one argument, the textfile, and has an optional argument for
how much data you want to sample from the unprocessed Wikipedia pages.

```
usage: main.py [-h] [--samples SAMPLES] datafile [datafile ...]

Evaluating Category Classification in Information Retrieval

positional arguments:
  datafile           the unprocessed cbor data file for Wikipedia pages

optional arguments:
  -h, --help         show this help message and exit
  --samples SAMPLES  number of samples to pull from the dataset
```

