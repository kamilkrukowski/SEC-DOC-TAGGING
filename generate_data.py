"""
    This file will downloads and parses financial data from the SEC EDGAR
    database for a list of companies specified in the "tickers.txt" file.
    The data is then processed and saved as a tokenizer
        and raw data file in the current directory.
"""
from tqdm.auto import tqdm


import EDGAR


# SETTINGS
DATA_DIR = 'data'
N_TIKRS = 15

loader = EDGAR.Downloader(data_dir=DATA_DIR)
parser = EDGAR.Parser(data_dir=DATA_DIR)

# List of companies to process
tikrs = parser.metadata.get_tikr_list()[:N_TIKRS]

tikrs = list(set(tikrs) - {'META'})

document_type = EDGAR.DocumentType('10q')

force = False
silent = True
remove = True

for tikr in tqdm(tikrs, desc='Processing...'):
    print(tikr)
    if len(parser.metadata._get_tikr(tikr)['submissions']) == 0:
        EDGAR.load_files(
            tikr, document_type='10q', force_remove_raw=remove,
            silent=silent, data_dir=DATA_DIR)

    annotated_subs = parser.get_annotated_submissions(tikr, silent=silent)

    if remove:
        extraneous = set(parser.metadata._get_tikr(
            tikr)['submissions']) - set(annotated_subs)
        for submission in extraneous:
            parser.metadata.offload_submission_file(tikr, submission)

    for submission in annotated_subs:
        fname = parser.metadata.get_10q_name(tikr, submission)
        if not parser.metadata.file_was_processed(tikr, submission, fname):
            EDGAR.load_files(
                tikr, document_type='10q', force_remove_raw=remove,
                silent=silent, data_dir=DATA_DIR)

        features = parser.featurize_file(
            tikr, submission, fname, force=force,
            silent=silent, remove_raw=remove)
