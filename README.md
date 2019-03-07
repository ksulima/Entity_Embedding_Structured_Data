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
<br>
As authors says, we map categorical variables in a function approximation problem into Euclidean spaces, which are the entity embeddings of the categorical variables. The mapping is learned by a neural network during the standard supervised training process. Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables.

It can rapidly generate great results on structured data without having to resort to feature engineering or apply domain specific knowledge. The technique is relatively straight forward, and simply involves turning the categorical variables into numbers and then assigning each value an embedding vector:

<img src="https://github.com/ksulima/Entity_Embedding_Structured_Data/blob/master/images/entity_schema.png" width="300" height="250">

## Project Steps

Below I summarize main steps of the project. If you are interested in full code and details, see notebook -> [Rossman](https://github.com/ksulima/Entity_Embedding_Structured_Data/blob/master/Rossman.ipynb)

### Data Cleaning / Feature Engineering
As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.<br>
Since data cleaning and feature engineering isn't the main goal in this project, I based this part on implementation from 'Practical Deep Learning for Coders' and fast.ai library by Jeremy Howard. You can find details [here.](https://www.fast.ai/2018/04/29/categorical-embeddings/)

### Durations
It is common when working with time series data to extract data that explains relationships across rows as opposed to columns, e.g.:
running averages, time until next event, time since last event.

### Create embeddings for catergorical variables

We incorporate Entity Embeddings concept to categorical variables in our dataset. 
We use the cardinality of each variable (that is, its number of unique values) to decide how large to make its embeddings. 
We calculate vector length with formula  ```min(50, (cardinality + 1)/2)) ```
```
 Variable name, cardinality -> vector length
 ('Store', 1116,) -> 50
 ('DayOfWeek', 8) -> 4
 ('Year', 4) -> 2
 ('Month', 13) -> 7
 ('Day', 32) -> 16
 ('StateHoliday', 3) -> 2
 ('CompetitionMonthsOpen', 26) -> 13
 ('Promo2Weeks', 27) -> 14
 ('StoreType', 5) -> 3
 ('Assortment', 4) -> 2
 ('PromoInterval', 4) -> 2
 ('CompetitionOpenSinceYear', 24) -> 12
 ('Promo2SinceYear', 9) -> 550
 ('State', 13) -> 7
 ('Week', 53) -> 27
 ('Events', 22) -> 11
 ('Promo_fw', 7) -> 4
 ('Promo_bw', 7) -> 4
 ('StateHoliday_fw', 4) -> 2
 ('StateHoliday_bw', 4) -> 2
 ('SchoolHoliday_fw', 9) -> 5
 ('SchoolHoliday_bw', 9) -> 5
```

### Build fully connected neural networks to predict sales

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_4 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_5 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_7 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_8 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_9 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_10 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_11 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_12 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_13 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_14 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_15 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_16 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_17 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_18 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_19 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_20 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_21 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_22 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 50)        55800       input_1[0][0]                    
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 1, 4)         32          input_2[0][0]                    
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 1, 2)         8           input_3[0][0]                    
__________________________________________________________________________________________________
embedding_4 (Embedding)         (None, 1, 7)         91          input_4[0][0]                    
__________________________________________________________________________________________________
embedding_5 (Embedding)         (None, 1, 16)        512         input_5[0][0]                    
__________________________________________________________________________________________________
embedding_6 (Embedding)         (None, 1, 2)         6           input_6[0][0]                    
__________________________________________________________________________________________________
embedding_7 (Embedding)         (None, 1, 13)        338         input_7[0][0]                    
__________________________________________________________________________________________________
embedding_8 (Embedding)         (None, 1, 14)        378         input_8[0][0]                    
__________________________________________________________________________________________________
embedding_9 (Embedding)         (None, 1, 3)         15          input_9[0][0]                    
__________________________________________________________________________________________________
embedding_10 (Embedding)        (None, 1, 2)         8           input_10[0][0]                   
__________________________________________________________________________________________________
embedding_11 (Embedding)        (None, 1, 2)         8           input_11[0][0]                   
__________________________________________________________________________________________________
embedding_12 (Embedding)        (None, 1, 12)        288         input_12[0][0]                   
__________________________________________________________________________________________________
embedding_13 (Embedding)        (None, 1, 5)         45          input_13[0][0]                   
__________________________________________________________________________________________________
embedding_14 (Embedding)        (None, 1, 7)         91          input_14[0][0]                   
__________________________________________________________________________________________________
embedding_15 (Embedding)        (None, 1, 27)        1431        input_15[0][0]                   
__________________________________________________________________________________________________
embedding_16 (Embedding)        (None, 1, 11)        242         input_16[0][0]                   
__________________________________________________________________________________________________
embedding_17 (Embedding)        (None, 1, 4)         28          input_17[0][0]                   
__________________________________________________________________________________________________
embedding_18 (Embedding)        (None, 1, 4)         28          input_18[0][0]                   
__________________________________________________________________________________________________
embedding_19 (Embedding)        (None, 1, 2)         8           input_19[0][0]                   
__________________________________________________________________________________________________
embedding_20 (Embedding)        (None, 1, 2)         8           input_20[0][0]                   
__________________________________________________________________________________________________
embedding_21 (Embedding)        (None, 1, 5)         45          input_21[0][0]                   
__________________________________________________________________________________________________
embedding_22 (Embedding)        (None, 1, 5)         45          input_22[0][0]                   
__________________________________________________________________________________________________
input_23 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_24 (InputLayer)           (None, 3)            0                                            
__________________________________________________________________________________________________
input_25 (InputLayer)           (None, 3)            0                                            
__________________________________________________________________________________________________
input_26 (InputLayer)           (None, 2)            0                                            
__________________________________________________________________________________________________
input_27 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_28 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_29 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_30 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_31 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_32 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
input_33 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 50)           0           embedding_1[0][0]                
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 4)            0           embedding_2[0][0]                
__________________________________________________________________________________________________
reshape_3 (Reshape)             (None, 2)            0           embedding_3[0][0]                
__________________________________________________________________________________________________
reshape_4 (Reshape)             (None, 7)            0           embedding_4[0][0]                
__________________________________________________________________________________________________
reshape_5 (Reshape)             (None, 16)           0           embedding_5[0][0]                
__________________________________________________________________________________________________
reshape_6 (Reshape)             (None, 2)            0           embedding_6[0][0]                
__________________________________________________________________________________________________
reshape_7 (Reshape)             (None, 13)           0           embedding_7[0][0]                
__________________________________________________________________________________________________
reshape_8 (Reshape)             (None, 14)           0           embedding_8[0][0]                
__________________________________________________________________________________________________
reshape_9 (Reshape)             (None, 3)            0           embedding_9[0][0]                
__________________________________________________________________________________________________
reshape_10 (Reshape)            (None, 2)            0           embedding_10[0][0]               
__________________________________________________________________________________________________
reshape_11 (Reshape)            (None, 2)            0           embedding_11[0][0]               
__________________________________________________________________________________________________
reshape_12 (Reshape)            (None, 12)           0           embedding_12[0][0]               
__________________________________________________________________________________________________
reshape_13 (Reshape)            (None, 5)            0           embedding_13[0][0]               
__________________________________________________________________________________________________
reshape_14 (Reshape)            (None, 7)            0           embedding_14[0][0]               
__________________________________________________________________________________________________
reshape_15 (Reshape)            (None, 27)           0           embedding_15[0][0]               
__________________________________________________________________________________________________
reshape_16 (Reshape)            (None, 11)           0           embedding_16[0][0]               
__________________________________________________________________________________________________
reshape_17 (Reshape)            (None, 4)            0           embedding_17[0][0]               
__________________________________________________________________________________________________
reshape_18 (Reshape)            (None, 4)            0           embedding_18[0][0]               
__________________________________________________________________________________________________
reshape_19 (Reshape)            (None, 2)            0           embedding_19[0][0]               
__________________________________________________________________________________________________
reshape_20 (Reshape)            (None, 2)            0           embedding_20[0][0]               
__________________________________________________________________________________________________
reshape_21 (Reshape)            (None, 5)            0           embedding_21[0][0]               
__________________________________________________________________________________________________
reshape_22 (Reshape)            (None, 5)            0           embedding_22[0][0]               
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            2           input_23[0][0]                   
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 3)            12          input_24[0][0]                   
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 3)            12          input_25[0][0]                   
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 2)            6           input_26[0][0]                   
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            2           input_27[0][0]                   
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 1)            2           input_28[0][0]                   
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            2           input_29[0][0]                   
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1)            2           input_30[0][0]                   
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            2           input_31[0][0]                   
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            2           input_32[0][0]                   
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1)            2           input_33[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 215)          0           reshape_1[0][0]                  
                                                                 reshape_2[0][0]                  
                                                                 reshape_3[0][0]                  
                                                                 reshape_4[0][0]                  
                                                                 reshape_5[0][0]                  
                                                                 reshape_6[0][0]                  
                                                                 reshape_7[0][0]                  
                                                                 reshape_8[0][0]                  
                                                                 reshape_9[0][0]                  
                                                                 reshape_10[0][0]                 
                                                                 reshape_11[0][0]                 
                                                                 reshape_12[0][0]                 
                                                                 reshape_13[0][0]                 
                                                                 reshape_14[0][0]                 
                                                                 reshape_15[0][0]                 
                                                                 reshape_16[0][0]                 
                                                                 reshape_17[0][0]                 
                                                                 reshape_18[0][0]                 
                                                                 reshape_19[0][0]                 
                                                                 reshape_20[0][0]                 
                                                                 reshape_21[0][0]                 
                                                                 reshape_22[0][0]                 
                                                                 dense_1[0][0]                    
                                                                 dense_2[0][0]                    
                                                                 dense_3[0][0]                    
                                                                 dense_4[0][0]                    
                                                                 dense_5[0][0]                    
                                                                 dense_6[0][0]                    
                                                                 dense_7[0][0]                    
                                                                 dense_8[0][0]                    
                                                                 dense_9[0][0]                    
                                                                 dense_10[0][0]                   
                                                                 dense_11[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 215)          0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1000)         216000      dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 500)          500500      dense_12[0][0]                   
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 1)            501         dense_13[0][0]                   
==================================================================================================
Total params: 776,502
Trainable params: 776,502
Non-trainable params: 0
```

As a final step I esemble 7 NN and average results for better predictions. I land with kaggle score 0.11125. 
<br>
**This reflects the leaderboard final standings on 35. positon.**


### Conclusion

Deep learning can be successfully applied to structured data. Entity Embeddings is a powerful technique to represent categorical variables in a form applicable for neaural networks. Entity Embeddings allows to capture information from categorical variables without extensively extended feature space e.g. by one-hot encoding.
