# Sparse Structure Learning via Graph Neural Networks for inductive document classification

<p align="center">
  <img src="TextSSL.png" />
</p>


## About data
We use the same benchmark datasets that are used in Yao, Mao, and Luo 2019, 
where we follow the same train/test splits and data preprocessing for MR, Ohsumed and 20NG datasets as Kim 2014; Yao, Mao, and Luo 2019. 
Thanks for their work.

For R8 and R52 datasets, they are only provided by a preprocessed version that lack punctuations and do not have explicit sample names. 
Since we use documents with sentence segmentation information to construct graph, we re-extract the data from original Reuters-21578 dataset.

You can download the dataset here: 
http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-115Cat.tar.gz

1. re-extract R8 and R52 datasets.
    ```
    python re-extract_data/mk_R8_R52.py
    ```

## Make graph dataset

1. create co-occurrence graph for datasets. 
    ```
    python ssl_make_graphs/create_cooc_document.py --raw_path SOURCEPATH --pre_path TARGETPATH --task DATASET --partition TRAINorTEST --window_size SIZE
    ```

2. construct in memory graph datsets.
    ```
    python ssl_make_graphs/PygDocsGraphDatset.py --raw_path SOURCEPATH --task DATASET 
    ```


## Reproduce

    python ssl_graphmodels/pyg_models/train_docs.py

