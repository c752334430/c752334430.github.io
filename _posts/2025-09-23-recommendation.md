---
title: 'Recommendation by Shusen'
date: 2025-09-22
permalink: /posts/2025/recommendation/
tags:
  - study
  - recommendation
---

## Retrieval

### ItemCF

Assumption: If user like $i_1$, and $i_1$ is similar to $i_2$, then user is likely to like $i_2$.

#### How is similarity calculated

Name the set of user like $i_k$ as $W_k$, then 

$$sim(i_1, i_2) = \frac{|W_1\cap W_2|}{\sqrt{|W_1|\cdot |W_2|}}$$

Which is cos similarity if we encode items based on one-hot encoding of user interactions.

#### What could go wrong

Say we have a community like wechat group, two irrelevant items can be liked by all the users in the group. To address this, we should weight less users from the same group. Generally, if users are unrelated but all like two items, then these two items should be more similar. This motivates the \bt{Swing} model. Name the user set that likes two items as $V$, for any two users $u_1,u_2\in V$, we estimate the probability of them coming from the same group using their liked-item overlap, $\sum_{u_1}\sum_{u_2} \frac{1}{\alpha + overlap(u_1,u_2)}$.

### Two tower model

#### Note

From DR paper: Despite their success in real world applications, vector-based methods with ANN or MIPS have two main disadvantages: (1)the inner product structure of user and item embeddings might not be sufficient to capture the complicated structure of user-item interactions 【10】. (2) ANN or MIPS is designed to approximate the learnt inner product model, not directly optimized for the user-item interaction data. (what does this mean?)

### Deep retrieval

[paper](https://arxiv.org/abs/2007.07203)

There are two mappings: User to path, and path to items. Here we have l layers where each layer contains K nodes. A path connects l nodes pick from all layers. Let x be user representation and path=[a,b,c], Model assumes $p(path|x)=p(a|x)\cdot p(b|a,x)\cdot p(c|a,b,x)$. It uses a NN with three components to compute it.

We train two parts: item representation using top related paths (here we don't want a path to map to too many items, so add a regularization), and the neural network model that estimate the relevance between user and path (path given user).

Inference: Given an user, use the NN to find top relevant paths with beam search, then from the mapping to get all items based on paths, finally use a small ranking model to cut the size if needed.

#### Note

1. It differs from two tower, by using paths instead of embeddings as the intermediate object. Also it aims to do multi-interest retrieval instead of single-interest in the two tower.
2. TT deprecated deep retrieval because there is a better model. 
3. Trinity: Syncretizing Multi-/Long-tail/Long-term Interests All in One

## Ranking