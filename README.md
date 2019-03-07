## Introduction

In this project my goal is to use neural networks to structured, time series data. I use method called "Entity Embeddings" to learn the representation of categorical features in multi-dimensional spaces. Such transformed data together with numerical features is easily consumed by neural network and it allows to train deep learning predictive model.

## Motivation

Nowadays most of the research in deep learning field is focused on unstructured data, like computer vision, natural language processing, where neural networks bring outstanding results comparing to others methods. Exploring deep learning to structured data is not in a academic spotlight, whereas lots of business problems and decisions are related to structured data. My plan is to apply neural networks to practical real-world problem.

## About dataset

I use a dataset from Kaggle [Rossman competiton.](https://www.kaggle.com/c/rossmann-store-sales) You can download all provided data together with external datasets put by participants from this [link](http://files.fast.ai/part2/lesson14/rossmann.tgz).

We are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales".
Dataset consists of 6 relational tables with both **categorical** and **numerical** variables. <br>The example data:<br><br>
<img src="https://github.com/ksulima/Entity_Embedding_Structured_Data/blob/master/images/tabular_data_sample.PNG" width="800" height="140">

## Entity Embeddings
Entity Embeddings was originally described by Cheng Guo and Felix Berkhahn in a [paper.](https://arxiv.org/abs/1604.06737)

The basic idea of using entity embeddings is to use a different set of dimension to represent a categorical set of data.
This approach allows for relationships between categories to be captured.
As authors says, we map categorical variables in a function approximation problem into Euclidean spaces, which are the entity embeddings of the categorical variables. The mapping is learned by a neural network during the standard supervised training process. Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables.

It can rapidly generate great results on structured data without having to resort to feature engineering or apply domain specific knowledge. The technique is relatively straight forward, and simply involves turning the categorical variables into numbers and then assigning each value an embedding vector:

<img src="https://github.com/ksulima/Entity_Embedding_Structured_Data/blob/master/images/entity_schema.PNG" width="400" height="300">

## Project Steps

Below I summarize main steps of the project. If you are interested in full code and details, see notebook -> [Rossman](https://github.com/ksulima/Entity_Embedding_Structured_Data/blob/master/Rossman.ipynb)


### Exporatory data summary
### Data Cleaning / Feature Engineering

Since data cleaning and feature engineering isn't the main goal in this project, I based this part on implementation from 'Practical Deep Learning for Coders' and fast.ai library by Jeremy Howard. You can find details [here.](https://www.fast.ai/2018/04/29/categorical-embeddings/)



### Durations
### Create features
### Fast-check random forest model
### Create embeddings for catergorical variables
### Build fully connected neural networks to predict sales



### Conclusion

Deep learning can be successfully applied to structured data. Entity Embeddings is a powerful technique to represent categorical variables in a form applicable for neaural networks. Entity Embeddings allows to capture information from categorical variables without extensively extended feature space e.g. by one-hot encoding.
