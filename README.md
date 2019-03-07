## Introduction

In this project my goal is to use neural networks to structured, time series data. I use Keras to implement method "Entity Embeddings" originally described by Cheng Guo and Felix Berkhahn in a [paper.](https://arxiv.org/abs/1604.06737)


### Motivation

Nowadays most of the research in deep learning field is focused on unstructured data, like computer vision, natural language processing, where neural networks bring outstanding results comparing to others methods. Exploring deep learning to structured data is not in a academic spotlight, whereas lots of business problems and decisions are related to structured data. My plan is to apply neural networks to practical real-world problem.

### Dataset description

I use a dataset from Kaggle [Rossman competiton.](https://www.kaggle.com/c/rossmann-store-sales) You can download all provided data together with external datasets put by participants from this [link](http://files.fast.ai/part2/lesson14/rossmann.tgz).

We are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales".
Dataset consists of 6 relational tables with both cateogrical and numerical variables. The example table:





Since data cleaning and feature engineering isn't the main goal in this project, I based this part on implementation from 'Practical Deep Learning for Coders' and fast.ai library by Jeremy Howard. You can find details [here.](https://www.fast.ai/2018/04/29/categorical-embeddings/)



### Project Steps:

- Exporatory data summary
- Data Cleaning / Feature Engineering
- Durations
- Create features
- Fast-check random forest model
- Create embeddings for catergorical variables
- Build fully connected neural networks to predict sales

Each step is described in main notebook [Rossman](https://github.com/ksulima/Entity_Embedding_Structured_Data/blob/master/Rossman.ipynb)


### Conclusion

Deep learning can be successfully applied to structured data. Entity Embeddings is a powerful technique to represent categorical variables in a form applicable for neaural networks. Entity Embeddings allows to capture information from categorical variables without extensively extended feature space e.g. by one-hot encoding.
