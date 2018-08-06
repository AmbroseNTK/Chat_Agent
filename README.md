# Hybrid Generative Model for learning Phrase Reprsentative human like conversation -- Chat_Agent 
*Asian University Machine Learning Camp - Jeju 2018*

# Overview
Basically,many emergency services,service company and ecommerce companies require helplines and people. But the catch here is… 
Emergency disaster /natural disaster requires people to not only assure them help but also to guide them,  pacify them, and at the same 
time provide back-up to the police or doctors with their current locations.  Now can all the police/doctors/people be available all the days for 24hrs???No! It’s not feasible, right? hence, I came up with this solution.How about an AI assistant like google or Alexa.  An AI which could exhibit emotions and can response to **consult as well as console people** with provided service specific answers   along with basic knowledge and intelligence of a human. Now this should pass the turning test of AI i.e. we all know an AI is   the best if it can pass the turning test, that is a human will interact with it not knowing that he is interacting with an AI and gets convinced that he is speaking or chatting to a human.

Table of Contents
=================
* [Overview](#overview)
* [Introduction](#introduction)
  * [Chat Agent](#chat-agent)
  * [Practical Demand meets solution](#practical-demand-meets-solution)
* [Motivation](#motivation)
* [Goal](#goal)
* [Prerequisite](#prerequisite)
* [Installation](#installation)
* [Graphical Representation](#graphical-representation)
* [Create Chat Agent](#create-chat-agent)
* [Getting Dataset](#getting-dataset)
* [Chat Data Structure](#chat-data-structure)
* [Chat Buffer and Insert Logic](#chat-buffer-and-insert-logic)
* [Building Database](#building-database)
* [Training the model](#training-the-model)
* [Interacting with Chat Agent](#interacting-with-chat-agent)


# Introduction

# Chat Agent?

Design of codes through machine learning’s NLP AND ARTIFICIAL INTELLIGENCE  THAT COULD BE A GENERIC INFORMATION RETRIVAL TO SPECIFIC INFORMATION

Level 1 : generic information retrieval fused with domain specific

Level 2 : generic information retrieval fused with guided information direction

Level 3 : generic information retrieval master artificial intelligence.

![chat agent](https://github.com/YangShyrMing/Chat_Agent/blob/master/s1.PNG)
# Practical Demand meets solution

Basically, how many emergency services, commercial services, service company and ecommerce companies require helplines and people??? Can they work 24hrs??? 
Let me ask you another question… How many first aid services, emergency services, helplines are there??? There are many, right!! But the catch here is… Emergency disaster /natural disaster requires people to not only assure them help but also to guide them, pacify them, and at the same time provide back-up to the police or doctors with their current locations. 
Now can all the police/doctors/people be available all the days for 24hrs???   No! It’s not feasible, right? hence, I came up with this solution. 

I am not talking about AI dominating world or dominating even jobs of people. But how about an AI assistant like google or Alexa. An AI which could exhibit emotions and can response to consult as well as console people with provided service specific answers along with basic knowledge and intelligence of a human. Now this should pass the turning test of AI i.e. we all know an AI is the best if it can pass the turing test, that is a human will interact with it not knowing that he is interacting with an AI and gets convinced that he is speaking or chatting to a human.

# Motivation

Research on data handling and sql and retrieval
21st century – bigdata and nosql – basically information retrieval – putting data into structured, semi-structured and unstructured  data
World is shifting to new CONCEPTS - a new technology of search and retrieval style of working.

# Goal

Generic information Retrieval Master **Artificial Intelligence**.

# Prerequisite

1. python 3.6
2. pip version 9

# Installation

1. tqdm
2. colorama
3. regex
4. Python-Levenshtein
5. requests

downloading can be done from the requirements file uploaded using the command **pip install -r requirements.txt**

# Getting Dataset

Dataset can be downloaded from
https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/?st=j9xmvats&sh=5843d18e

# Graphical Representation
Step 1: WHAT DO WE DO when we have some data for ingestion???we leverage nlp to make our own dataset for "the end better dataset to train"

![img1](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej1.jpeg)

Step 2: we use nlp concepts in step by step order for better Results And prediction corpus building

![img2](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej2.jpeg)

Step 3: we focus ON THE FOLLOWING to build a chatbot

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej3.jpeg)

Step 4:Lexical analysis to break down THE SENTENCE to WORDS AND keep only meaningful And parse it to next…..

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej4.jpeg)

Step 4b: syntactic analysis is to make sense of what the words mean, then sort and order them to make sense for everyone, as well as help the machine learning to be sensible

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej5.jpeg)

Step 5:our model is not only step by step, but a fuse of mixing in enumeration style, thus we get more efficient result

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej7.jpeg)

Step 6: what parameters do we use ???We combine into next phase of concepts, entities and keywords

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej8.jpeg)

Step 7: forming our final search, like to be analyzed in neural networks training
![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej9.jpeg)

Step 8: Finally to make sense, not only to us but also, help in fuse corpus building in advance form in starting so... training of understanding will be better.. once nlp completes and pass to neural networks

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej10.jpeg)

Step 9: the neural networks forms a style of question and answer on its own...as we make sense to its understanding

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej11.jpeg)

Step 9b: Look!! how a sentence breaks down to understand and reply

![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej12.jpeg)

Step 9c:Using our style of break down to further answer close to human
![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej13.jpeg)

Step 9d:When passed to neural networks to train.. it self scores all the answer And relevant answers it streams down to pull the right answer based on score
![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej14.jpeg)

Step 10:Final Result
![img3](https://github.com/YangShyrMing/Chat_Agent/blob/master/tej15.jpeg)

# Create Chat Agent
![](https://github.com/YangShyrMing/Chat_Agent/blob/master/git1.PNG)

# Chat Data Structure
There is alot of unwanted information in the data we downloaded. So it must be cleaned. Thus, we perform data preprocessing  
https://github.com/YangShyrMing/Chat_Agent/blob/master/t.PNG

# Chat Buffer and Insert Logic
![](https://github.com/YangShyrMing/Chat_Agent/blob/master/t1.PNG)
![](https://github.com/YangShyrMing/Chat_Agent/blob/master/t3.PNG)

# Building Database
![](https://github.com/YangShyrMing/Chat_Agent/blob/master/t4.PNG)

# Training the model
We will get a set of question and answers in the form of 2 text files as text.to and text.from 
![](https://github.com/YangShyrMing/Chat_Agent/blob/master/t5.PNG)
 
# Interacting with Chat Agent
![](https://github.com/YangShyrMing/Chat_Agent/blob/master/op.PNG)
 

# References

https://medium.com/@erikhallstrm/hello-world-tensorflow-649b15aed18c

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/




