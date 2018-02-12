# HIEPT
Python implementation for HIEPT (Hashing User Intrinsic and Extrinsic Preference for Next Topic Prediction)

## Requirements
- python >= 2.7
- tensorflow >= 1.3.0
- numpy >= 1.14.0
- bitarray >= 0.8.1
- matplotlib >= 1.5.11

## Datasets
The datasets used in this project are from four main social media platforms, including twitter, weibo.com, zhihu.com, douban.com, and Twitter.com. Due to the privacy concern, we do not provide the raw data (available upon request). However, the intermediate data, such as two types of embedding vectors (tweets and social networks) can be found in the ` data`  folder.

The ` data` folder consists of four datasets, each having three subfolders:
- graph_embedding: embedding vectors for social networks, include [DeepWalk](https://github.com/phanein/deepwalk), [LINE](https://github.com/tangjianpku/LINE), [node2vec](https://github.com/aditya-grover/node2vec), [CANE](https://github.com/thunlp/cane), [TADW](https://github.com/thunlp/TADW),[struc2vec](https://github.com/leoribeiro/struc2vec)
- topics: topics that users have joined to discuss, each line represents a topic and every value in a line is a userid
- tweets_embedding: embedding vectors for user tweets, we use [gensim](https://radimrehurek.com/gensim/models/doc2vec.html) to implement [doc2vec](https://arxiv.org/pdf/1405.4053v2.pdf)

## Usage
To run HIEPT, first clone the project to your python IDE (eg:Pycharm), then run the `main.py`.
>Note: you need to install the required libs.

## Configuration
Change the configuration in `src/config.py`:
- data_set: specify which dataset you want to run. Possible values are:`douban`, `weiobo`, `twitter`, `zhihu`
- ne_type: specify the network embedding type. Possible values are:`node2vec`, `line`, `tadw`, `deepwalk`, `stuct2vec`, `cane`

You can find more configuration parameters in `src/config.py`
