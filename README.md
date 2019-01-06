## Introduction

In this project my goal is to use neural networks to structured, time series data. I use Keras to implement method "Entity Embeddings" originally described by Cheng Guo and Felix Berkhahn in a [paper.](https://arxiv.org/abs/1604.06737)


### Motivation

Nowadays most of the research in deep learning field is focused on unstructed data, like computer vision, natural language processing, where neaural networks bring outstanding results comparing to others methods. Exploring deep learning to structured data is not in a academic spotlight, whereas lots of business problems and decisions are related to structured data. My plan is to apply neural networks to practical real-world problem.

### Dataset description

I use a dataset from Kaggle competiton. You can download all provided data together with external datasets put by participants from this [link.](http://files.fast.ai/part2/lesson14/rossmann.tgz)
Since data cleaning and feature engineering is not my main goal in this project, I based this part of implementation from 'Practical Deep Learning for Coders' and fast.ai library by Jeremy Howard. You can find details [here](https://www.fast.ai/2018/04/29/categorical-embeddings/)



## Project Plan:

- Exporatory data summary
- Data Cleaning / Feature Engineering
- Durations
- Create features
- Fast-check random forest model
- Create embeddings for catergorical variables
