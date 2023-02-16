"""
     a Python script that downloads and processes
     financial data from the SEC EDGAR database based
     on companies specified in the "tickers.txt" file,
     and then creates a list of label in order from most
     frequent to least frequent.
"""

import os
import time
import itertools
import argparse


from tqdm.auto import tqdm
from secedgar import FilingType
import numpy as np

import EDGAR

# Command line magic for common use case to regenerate dataset
#   --force to overwrite outdated local files
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('-nflx', '--demo', action='store_true')
parser.add_argument(
    '-o', '--output_file',
    type=str, help='Output filename', default='labels.txt')
parser.add_argument(
    '-p', '--min_occurrence_perc',
    type=float, default=0, help='Minimum occurrence percentage')
parser.add_argument(
    '-c', '--min_occurrence_count',
    type=int, default=20, help='Minimum occurrence count')
parser.add_argument(
    '-n', '--nonnumeric', action='store_true',
    help='remove not nonnumeric label')
args = parser.parse_args()

# SETTINGS
PREPROCESS_PIPE_NAME = 'DEFAULT'
DATA_DIR = 'data'


def generate_label(
        raw_data,
        min_occur_perc,
        min_occur_count,
        file_path):
    # i is the data in document
    # j is the (text, list of list labels)
    # k is the list of list labels
    label_data = [k[0] for k in itertools.chain.from_iterable(
            [j[1] for j in itertools.chain.from_iterable(
                    [i[0] for i in raw_data])])]
    all_labels_count = len(label_data)
    all_labels, counts = np.unique(label_data, return_counts=True)
    reindexing = list(reversed(np.argsort(counts)))
    # Create a dictionary of words and their counts
    label_counts = dict(zip(all_labels[reindexing], counts[reindexing]))
    # Create a list of words that meet the criteria
    selected_labels = [label for label, count in label_counts.items()
                       if count >= min_occur_count and
                       count / all_labels_count >= min_occur_perc]

    # Remove all company specific systems predicted
    kept_systems = {'dei', 'us-gaap'}
    selected_labels = [i for i in selected_labels if i.split(':')[
        0] in kept_systems]
    np.savetxt(file_path, [label for label in selected_labels], fmt='%s')


# initialization
loader = EDGAR.downloader(data_dir=DATA_DIR)
metadata = EDGAR.metadata(data_dir=DATA_DIR)
parser = EDGAR.parser(data_dir=DATA_DIR)


# List of companies to process
tikrs = open(os.path.join('tickers.txt')).read().strip()
tikrs = [i.split(',')[0].lower() for i in tikrs.split('\n')]
if args.demo:
    tikrs = ['nflx']

for tikr in tikrs:
    loader.metadata.load_tikr_metadata(tikr)


def download_tikrs(tikrs):
    to_download = []
    if args.force:
        print('Forcing Downloads...')
        for tikr in tikrs:
            to_download.append(tikr)
    else:
        # Download missing files
        for tikr in tikrs:
            if not loader._is_downloaded(tikr):
                to_download.append(tikr)

    if len(to_download) != 0:
        for tikr in tqdm(to_download, desc='Downloading', leave=False):
            loader.query_server(tikr, force=args.force,
                                filing_type=FilingType.FILING_10Q)
            time.sleep(5)


download_tikrs(tikrs)

raw_data = list()
label_map = set()
for tikr in tikrs:
    # Unpack downloaded files into relevant directories
    loader.unpack_bulk(
        tikr,
        loading_bar=True,
        force=args.force,
        complete=False,
        document_type='10-Q',
        desc=f'{tikr} :Inflating HTM')
    annotated_docs = parser.get_annotated_submissions(tikr, silent=True)

    if (args.demo):
        annotated_docs = [annotated_docs[0]]

    # to process the documents and extract relevant information.
    for doc in annotated_docs:
        fname = metadata.get_10q_name(tikr, doc)
        features = parser.featurize_file(
            tikr, doc, fname, force=args.force, silent=True)
        found_indices = np.unique([int(i) for i in features['found_index']])
        # Structure: Text str, Labels dict, labelled bool
        data = {i: {'text': None, 'labels': dict(), 'is_annotated': False,
                'in_table': False, 'page_number': 0} for i in found_indices}

        for i in range(len(features)):
            i = features.iloc[i, :]
            if args.nonnumeric and i['anno_ix_type'] != 'ix:nonnumeric':
                continue
            d = data[i['found_index']]
            # Skip documents which are NOT annotated
            if i['in_table']:
                d['in_table'] = True
            if i['is_annotated']:
                d['is_annotated'] = True

            d['page_number'] = i['page_number']
            if d['text'] is None:
                """
                x is a list with length of 2. Items in the list are:
                    1. value: the text value of the annotated label (e.g. 10-Q)
                    2. neighboring text: the text on the given page.
                y is a list with
                    ['name', 'id', 'contextref',
                        'decimals', 'format', 'unitref']
                """
                d['text'] = i['span_text']

            if i['anno_index'] is not None:
                d['labels'][i['anno_index']] = []
                for _attr in ['name', 'id', 'contextref', 'decimals']:
                    d['labels'][i['anno_index']].append(i['anno_' + _attr])

        doc_data = []
        for i in data:
            # This checks for the all the element on a page. Only add element
            # that has labels to the training set.
            if data[i]['in_table']:
                continue
            if not data[i]['is_annotated']:
                continue
            d = data[i]
            labels = list(d['labels'].values())

            for label in labels:
                label_map = label_map.union({label[0]})

            # Data format: (x,y) where x refers to training features (present
            # for unnannotated docs), and y refers to labels to predict
            doc_data.append((d['text'], labels))

        raw_data.append([doc_data, doc, tikr])

label_map = {y: i for i, y in enumerate(label_map)}

# saves the raw data
vocab_dir = os.path.join(metadata.data_dir, 'dataloader_cache')
out_dir = os.path.join(vocab_dir, PREPROCESS_PIPE_NAME)
if not os.path.exists(out_dir):
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    os.mkdir(out_dir)
np.savetxt(os.path.join(out_dir, 'all_possible_labels.txt'),
           [key for key in label_map], fmt='%s')

generate_label(
    raw_data,
    args.min_occurrence_perc,
    args.min_occurrence_count,
    args.output_file)
