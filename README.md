# SEC-DOC-TAGGING

## Dependencies

We use [EDGAR-DOC-PARSER](https://kamilkrukowski.github.io/EDGAR-DOC-PARSER) to create a dataset.
It is currently available on the PyPi Test Server
```
  pip install -i https://test.pypi.org/simple/ EDGAR-Document-Parser
```


### Conda

To create conda environment:
```
conda create -n sectagging -c conda-forge -c anaconda python=3.10 pytorch scipy numpy selenium=4.5.0 pyyaml chardet requests lxml scikit-learn matplotlib evaluate transformers datasets
conda activate sectagging
```
