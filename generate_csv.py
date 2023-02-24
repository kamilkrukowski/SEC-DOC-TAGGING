"""
    This script downloads and processes financial data for a list of companies
    specified in the "tickers.txt" file, then creat and save
    label list. The script accepts command-line arguments such
    as '-f' to force overwriting of outdated local files
    and '-o' to specify an output file name.
"""

import os
import itertools
import argparse


from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import EDGAR


# SETTINGS
MAX_SENTENCE_LENGTH = 200
PREPROCESS_PIPE_NAME = 'DEFAULT'
SPARSE_WEIGHT = 0.5
MIN_OCCUR_PERC = 0
MIN_OCCUR_COUNT = 20
VOCAB_LENGTH = 12000
DATA_DIR = 'data'
SILENT = True
N_TIKRS = 5

# Command line magic for common use case to regenerate dataset
#   --force to overwrite outdated local files
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('-nflx', '--demo', action='store_true')
parser.add_argument(
    '-m', '--max_length', type=float, default=1000,
    help='Maximum span text for each label')
parser.add_argument(
    '-l', '--labels', type=str, default='labels.txt',
    help='path to the label file')
parser.add_argument(
    '-o', '--output_file',
    type=str, help='Output filename', default='data.csv')
parser.add_argument(
    '-n', '--nonnumeric', action='store_true',
    help='remove not nonnumeric label')
args = parser.parse_args()

loader = EDGAR.Downloader(data_dir=DATA_DIR)
metadata = EDGAR.Metadata(data_dir=DATA_DIR)
parser = EDGAR.Parser(data_dir=DATA_DIR)

# List of companies to process
tikrs = parser.metadata.get_tikr_list()[:N_TIKRS]
if args.demo:
    tikrs = ['nflx']


def change_digit_to_alphanumeric(text):
    for alph in '0123456789':
        text = text.replace(alph, '0')
    return text


raw_data = list()
label_map = set()
for tikr in tqdm(tikrs, desc='Processing...'):

    if len(parser.metadata._get_tikr(tikr)['submissions']) == 0:
        EDGAR.load_files(
            tikr, document_type='10q', force_remove_raw=False,
            silent=SILENT, data_dir=DATA_DIR)
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
                # d['text'] = i['span_text']
                d['text'] = i['anno_text']

            if i['anno_index'] is not None:
                d['labels'][i['anno_index']] = []
                attrs = ['name', 'id', 'contextref', 'decimals', 'ix_type']
                for _attr in attrs:
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

# if label is given then use the given label
# otherwise generate a label list
if (args.labels is not None
   and os.path.isfile(args.labels)):
    with open(args.labels, 'r') as f:
        selected_labels = f.read().splitlines()
    print(f'Label file {args.labels} found')
else:
    print('file not found, generate new label.txt')
    # i is the data in document
    # j is the (text, list of list labels)
    # k is the list of list labels
    label_data = [k[0] for k in itertools.chain.from_iterable(
        [j[1] for j in itertools.chain.from_iterable(
            [i[0] for i in raw_data])])]
    all_labels_count = len(label_data)
    all_labels, counts = np.unique(label_data, return_counts=True)
    # sort the data based on its counts from most frequent
    # to least frequent
    reindexing = list(reversed(np.argsort(counts)))
    # Create a dictionary of words and their counts
    label_counts = dict(zip(all_labels[reindexing], counts[reindexing]))
    # Create a list of words that meet the criteria
    selected_labels = [label for label, count in label_counts.items()
                       if count >= MIN_OCCUR_COUNT and
                       count / all_labels_count >= MIN_OCCUR_PERC]

    # Remove all company specific systems predicted
    kept_systems = {'dei', 'us-gaap'}
    selected_labels = [i for i in selected_labels if i.split(':')[
        0] in kept_systems]
    np.savetxt('labels.txt', [
               label for label in selected_labels], fmt='%s')

# Generate span text and labels
label_text = {y: [None] * args.max_length for y in selected_labels}
label_size = {y: 0 for y in selected_labels}
for document in raw_data:
    elems, doc_id, tikr = document
    for elem in elems:
        # inputs.append(tokenizer(elem[0],
        # return_tensors='pt', truncation=True))
        for label in elem[1]:
            if (label[0] in selected_labels
               and label_size[label[0]] < args.max_length):
                curr_index = label_size[label[0]]
                label_text[label[0]][curr_index] = elem[0]
                label_size[label[0]] += 1


df = pd.DataFrame.from_dict(label_text)
df.to_csv(args.output_file)
