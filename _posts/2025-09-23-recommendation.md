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

### UserCF

Assumption: If user A like $i$, user B is similar to user A, then user B may also like $i$.

#### How is similarity calculated

Check 1. the items liked, shared, saved by the users; 2. the authors subscribed by the users.

However, popular items are liked by many users, thus they should contribute less in the similarity computation. So let $n_l$ be the number of users like item $l$, and name user $u_k$ like item set $J_k$, then

$$ sim(u_1, u_2)= {\sum_{l\in J_1\cap J_2} 1/\log (1+n_l) \over \sqrt{|J_1|\cdot |J_2|}}$$

### Two tower model

#### Negative sampling

Easy negative: Samples not retrieved previously (uniform samples or in-batch samples). The probability that a sample  being used as negative should be propotional to $click\_num ^{0.75}$. (use log Q correction)

Hard negative: Samples retrieved but not selected by the ranking models.

In pactice we can mix easy and hard samples. **But You can not use samples not clicked as negatives!** While this is a conclusion from practcie, the hypothesis is that retrieval is only responsible for filter out items an user is not going to like. It is not responsible for **ranking** if an item is very interesting to the user or not.

#### Serving

Pre-compute embeddings for items and save in vector database. 

**But we should not pre-compute embeddings for users** -- why? Cause user interest changes over time, while item property is more stable, does not change easily.

#### Model update

1. Full update/load: Every day, train the previous model with one day data for 1 epoch, release the new user tower and item embeddings. Random shuffle to avoid bias with time.

2. Incremental udpate: Online learning. User interest can be tracked faster. We collect realtime data (hour or even minute level), and refresh ID embedding only (do not change other params in the NN). We cannot only rely on this as user behavior changes periodically.

In practice we should use both updates. changes made by the incremental update should not be used in the next full update.

#### Self-supervised learning

Two tower can learn better embeddings for popular items but not unpopular items. So we can do data augmentation for uniformly sampled items in batch: 1. random transform item $i_1, i_2$, ... using two methods into $i_1 '$, $i_1 ''$, $i_2 '$, $i_2 ''$, .... Then train the item towers so $i_k '$, $i_k ''$ are similar, while $i_k '$, any $i_j ''$ are different (cross entropy). Add this into the total loss function.

Random transformation includes: 

1. Random mask that remove info of a discrete field e.g., mask out category.

2. Dropout that only remove part of a discrete field, e.g., randomly remove half of the categories

3. Complementary, randomly split features into two groups, transform 1 only use features from group 1, and transform 2 use features from group 2.

4. Mask out a group of highly-correlated features. For example (preferred gender = female) and  (categoriy = beauty) have high correlation, we measure using mutual information. Say we have k features, compute $k\times k$ mutual information matrix. Then randomly select a seed feature, get $k/2$ features that are most related to the seed, mask all of them. **This method usually outperform other transformations**, but it's hard to implement and maintain (e.g., need to recompute mutual info for new features).

**In practice this method shows very strong perf on low-popular and new items**

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

### Other retrievals

#### Author based

Subscribed authors, authors you interacted with, similar authors.

#### Geo based

Geohash retrieval and city-based retrieval. They only maintain most popular items in the area, no personalization.

#### Cache based

If an item passed ranking (top 50) previously but not picked by random selection in re-rank (for diversity), then cache them for the retrieval next time.

### Impression Filter

If user has interacted with an item in history, then we filter them out after retrieval. This stage may not be necessary for long content like youtube videos.

Filter in Brute force can waste time (thousands interacted items * thousands retrieved items). So we use bloom filter instead.

## Ranking

Thousands of items -> light ranker -> hundreds of items -> heavy ranker -> rerank.

### Multi-object model

Ranking models estimate impression rate, like rate, save rate and share rate at once. It takes user features, item features, stat features (user click num in last 30 days...) and staging features (day of time...). 

Features are sent to a large NN (shared bottom) then branch into different MLP+sigmoid to predict all rates. 

#### Label imbalance

Usually we have more negative samples compared to positive ones. Solution can be down sampling on negatives.

#### Correction on estimations

Due to the down sampling on negatives, there will be a bias on positive results. Say the down sample rate is $\alpha$, number of real positives $n_+$ and negatives $n_-$, the real click rate is $p_{true}={n_+\over n_+ + n_-}$, the predicted click rate $p_{pred}$ will multiply $\alpha$ with $n_-$. So we should adjust it: 
$$ p_{true}={\alpha \cdot p_{pred} \over (1-p_{pred})+ \alpha\cdot p_{pred}} $$

### Multi-gate mixture of experts (MMoE)

Use $k$ multiple NNs (experts) instead of one NN to process the features, e.g., $k=8$. Also use NN + softmax ot predict probabilities $p_1^i,...p_k^i$, for $i=1,2,...$. Wegihted sum of outputs from experts will be used to estimate different rate.

#### Polarization

In practice, $p$ vector is likely to be polarized to one expert, thus lose the advance of having multiple experts. To address this, we apply dropout to the output of softmax (randomly drop 10% of the values).

#### Performance

People does not always see improvement with MMoE.

### Aggregation of estimated scores

#### Weighted sum

1. Simple weighted sum, $p_{click}+w_1\cdot p_{like}+...$

2. click rate multiples weighted sum, $p_{click}\cdot (1+w_1\cdot p_{like}+...)$

3. Let $p_{time}$ be the estimated time of watching, $(1+w_1\cdot p_{time})^{\alpha_1}\cdot (1+w_2\cdot p_{like})^{\alpha_2}...$

4. Rank items based on metrics, e.g., by $p_{time}$ we get item rank $r_{time}$, aggregate the ranking instead of scores as ${w_1\over r_{time}^{\alpha_1}+\beta_1} +{w_2\over r_{click}^{\alpha_2}+\beta_2}$

### Video

#### Metrics

1. Time of watching $t$. When training, the objective is to predict $t\over 1+t$ with a sigmoid output from NN.

2. Completion rate. We can use regression or binary classification to model. There is always bias towards short videos. So we need to adjust based on the length of the video.

### Features

#### User profile

1. User ID (embedding)
2. Biographical inf
3. Account info
4. Interested tags..

#### Item profile

1. Item ID
2. Published time
3. Geohash, location
4. Tag, category...
5. Number of words, iamges, resolution, tags...
6. Based on algorithms (pretrained CV and NLP models), we can get amount of useful information, beauty of images...

#### User stat

1. Last $k$ days/hours, the number of impressions, clicks...
2. Bucketize by interacted post types (text only or image or video)
3. Bucketize by interacted post categories

#### Item stat

1. Last $k$ days/hours, the number of impressions, clicks...
2. Bucketize by interacted user biographics...
3. Author features, number of follows, likes...

#### Context

1. Geohash, city
2. Timestamp
3. Phone, OS, hardware info

#### Feature engineering

1. Embedding for discrete features
2. Bucketize, normalize or transformation like $log(1+x)$ for continuous features
3. Many features cannot cover all users, items. How to boost coverage for important features, and how to deal with missing values

#### Service

For one request, we only retrieve features for one user, but features for thousands of items. So it's recommended to use light weighted embeddings/features for items to reduce the load. 

Profiles are usually static, especially for items. While stats are very dynamic, so need more frequent updates.

### Light ranker

#### Three tower model

Add an additional tower for stat features and crossed features. Model cocnate and cross three tower results and send to different MLP + sigmoid to estimate business metrics. The major computation comes from the upper layer (MLPs).

User tower can be large. Item tower can be relatively large, as item profile doesnt change frequently so we can use cache. Cross tower has to be small as it needs $n$ inference in realtime for $n$ items.

## Cross feature

### Factorized machine

Linear model with cross features

$$ p = b+\sum _{i=1}^d w_i\cdot x_i + \sum _{i=1}^d\sum _{j=i+1}^d u_{ij}x_ix_j$$

It has $O(d^2)$ parameters.

Factorized Machine uses $V\cdot V^T$ to approximate $U$, where $V\in R^{n\times k}$:

$$ p = b+\sum _{i=1}^d w_i\cdot x_i + \sum _{i=1}^d\sum _{j=i+1}^d (v_i^T v_j)x_ix_j$$

So the number of parameters is reduced to $O(dk)$. It brings less computation and prevent overfit.

**Note**: less used nowadays.

### DCN

It can be used in most models of retrieval and ranking.

#### Cross layer

Given initial input vector $x_0$, at layer $i$, the current input $x_i$, it will go through a MLP and transform to $y_i$. We do Hadamard product between $x_0$ and $y_i$ to get $z_i$. Finally $x_{i+1}=x_i + z_i$. Equivalently, we can write $x_{i+1}=x_0\circ \sigma [W\cdot x_i +b] + x_i$

#### Cross network

Stack multiple cross layers together. 

#### Deep and Cross Network

Send features to both cross network and MLP, concat output and send to another MLP.

### LHUC

Heavy ranker only. Kuaishou used initially and called it PPNet.

Casually define: 

f(Iteam, User) = Hadamard (Iteam -> MLP , User -> MLP)

The LHUC structure is $f(f(Item, User), User)$.

### SENet

Say an item has $m-1$ feature embeddings from different discrete features, together with the embedding of user id, each of dimension $k$ (this can actually be different, as we will do average pooling). Call this embedding matrix $M$. We learn a importance factor for each embedding, by: 1. Take average pooling for each vector ($R^{m\times 1}$), 2. Use FC + relu + FC + sigmoid to compress ($R^{m/r\times 1}$) then expand ($R^{m\times 1}$), 3. row-wise multiple the output with $M$.

Intuitively, this is weighted sum of discrete feature embeddings.

#### Functions for crossing

1. Dot product

2. Hadamard product

3. Bilinear Cross for both products: 
- $f_{ij} = x^T_i\cdot W_{ij} \cdot x_j$, there can be $m^2/2$ parameter matrices, each $d\times d$. So we only should select some pairs of features that are important.
- $f_{ij} = x_i\circ [W_{ij} \cdot x_j]$

#### FiBiNet

Take embeddings of discrete features, do three things: 1. concat, 2. SENet (+ bilinear), 3. Bilinear cross. Concat these three and continuous features, send to NN.

**Improvement observed in practice**

## User interaction sequence

### Last N

User behavior history, specifically, the last N items the user interacted with. We can take the embeddings of these items (ID, categories, etc) and take average, use that as an user feature. We can use this in two tower, three towers, and models in heavy ranker.

We can seperate item sequences of different interactions.

### DIN

Instead of average, people nowadays use attention. Weights are calculated by the similarity between candidate item and the last N items. This is used in heavy ranker. 

### SIM

- The drawback of DIN: can only record the last hundreds of items, otherwise the computation is too heavy. Therefore DIN will forget long term info.

SIM instead look at thousands of items, search for only say one hundred most similar items. These top K items are send to attention.

1. How to search: a. hard search by rules like category; b. soft search, embedding similarity, better in perf

2. attention: Add time of interaction (bucketized) as additional embedding (positional encoding). This is important for SIM but not DIN, as SIM look long-term interest, thus the important can be very different.

**Improvement observed in practice**

## Rerank

### Diversity
Post processing happens in both light and heavy rankers. Maintain diversity in both stages are important.

#### Similarity

We can use categories of items to compute the similarity. It can be weighted sum of indicators of different category levels. Or we can compute the similarity based on embeddings, in practice we do not use embeddings learned from two-tower. Instead it's better to use NLP + CV models like CLIP.

### MMR (Maximal marginal relevance)
Let the set of selected items be $S$, the set of items not selected as $R, $\text{reward}_i$ be the score of item i from heavy ranker, the marginal relevance score is:

$$ MR_i = \theta\cdot \text{reward}_i - (1-\theta)\cdot \max_{j\in S} sim(i,j) $$

We pick the item from $R$ with the largest MR and add it to $S$.

Con of MMR: when S is large, there is almsot laways an item similar to every candidcate.
- To address this, we apply a slide window, say of size 10, on the recently added items in S. We only care about the similarities with items in the window.

### Diversity constraints based on business logics

Examples from RedNote:

1. No more than k consecutive notes of the same type
2. At most one type of note (e.g. promo note, whose reward is usually multiplied by a boost value) in every k notes
3. At most k notes from a type in the first t notes (e.g., in the first 4 notes that forms the first loaded page, no more than 1 comercial)

To combine MMR with rules, we use rules to filter out unqualified notes, then pick by MR.

### DPP (determinantal point process)

Quantify item diversity as $det(V^TV)=vol(\text{space defined by V})^2$, where $V$ is item embedding matrix. To solve this approximately, DPP uses greedy to find for each step an item that maximize reward plus incremental det. Here the incremental det is calculated by chelosky decomposition. $det(V^TV) = \prod l_{ii}^2$, where $l_{ii}$ is i-th diagonal element in the lower triangle matrix. Note adding one row in V does not greatly change the result of Chelosky decomposition, see [note in chinese](https://github.com/wangshusen/RecommenderSystem/blob/main/Notes/06_Rerank.pdf)

## Cold start

### Metrics

#### Author side

1. penetration rate = num of users publishing / DAU
2. avg publish = num of notes published today / DAU

Observation: The more impressions got for a new note, the earlier first impression and interaction happens, the more authors get motivated to publish.

#### User side

1. Click rate, interaction rate of new notes. Problem: popular notes create bias, so we should bucketize / split notes by num of impressions, and focus on notes with low impressions.

2. Overall metrics, like time spent, DAU, MAU etc. Works on cold start usually decrease these metrics.

#### Content side

1. hot notes rate: the percentage of notes that receive 1000+ clicks in the first 30 days. 

### Retrieval

#### challenges

Due to the lack of interactions, we cannot use itemCF as the similarity is computed based on the sets of interacted users. We also cannot directly use two tower models, given the embedding of the item and the ID embedding of the item are not meaningful.

#### Two tower model

1. Use a learnable default embedding for all new items, until the item reach enough interactions.

2. Use averaged embeddings from top k similar items (that are with high impression), measured by NLP/CV methods.

3. Use multiple retrieval pools, bucketized by item live, e.g., 1 hour items, 6 hour items...

#### Category based

Maintain for each user a profile, depicts the item categories the user likes.

System maintain an index from category to item list ordered by time, when requested, return the top k items.

**Other simple logics like keywords based**

Cons:
1. Only works for very new items, e.g., item will not be retrieved after hours of publishing.

2. Not very personalized. 

### Clustering based

Use pretrained NN to embed items based on content. Use k-means to build say 1000 clusters. For each cluster, build index to items ordered by time.

On the user side, use its last N interacted items as seed items, get their embeddings and find the most similar cluster. Retrieve most recent m items from the cluster.

#### Similarity model

Use BERT, CNN etc. plus FC. Train with positive and negative samples. Positive samples can come from manual labeling, or picked based on category / itemCF. Negative sample can be uniformly sampled from the universe, as long as the item quality is good (has enough image/text, and with high quality).

### Look alike

We call users interacted with a note the seed users. Find similar user. E.g., UserCF, user embedding

One approach is to take the average embedding of the users interacted with the item. This average embedding need to be maintained in minute level. When a new user request comes in, find average embeddings that are similar to the user embeddings.

### Traffic control

#### Purpose

1. Advocate publishing new posts
2. Explore to find high-quality posts

#### How to do

1. Directly add new items in recommended result. (out-dated method)
2. Boost: In ranking stage, add or multiple some factor to new itmes. This happens during light ranking and rerank, where items are filtered. **Con: very sensitive to parameter control**
3. Gaurantee the traffic via boost. E.g., make sure new item get impressed at least 100 times, one way to do this is deterministically increase the multiple factor when an item has long published time but low impression count. 
4. Personalized  boost. Traffice guarantee set differently for items of different quality, or from different authors. For example, if an item is from a good author, we may guarantee 200 impression times instead of 100 times.

#### Challenges

Boost does not guarantee the traffic of an item, even if we dynamically change the boost factor. This can due to some weakness in retrieval or ranking, or insufficient parameter tuning. Consistently changed online system also provide a very unstable environment so the goal of traffic control cannot always be achieved.

Second, we cannot naively increase the boost factor to new items: it may be recommended to irrelevant users thus decrease the rate.

### AB test for cold start

#### Normal AB test for a recommendation system

- do not classify items, we only sample a set of users to apply the new apporach. Compare user side metrics (see Metrics in this section).

#### For cold start

- Compare both user side and author side metrics
- Due to traffic control, AB test may see worse user side consuming metrics than real deployment. Reason: Say we guarantee 100 impressions for new items, then selected user set will see way more new items in AB test, than the whole user set in deployment, due to the cardinality difference.
- Author side metrics are hard to test: **incorrect settings:** Split new notes by authors into two groups and compare results, (i) seperate recommend channels for new items and old items. In this case, experimental group may take away traffic from the other group and show better author side metrics. (ii) allow new and old items compete, then the cardinality difference in AB test can cause (say traffic control propotion) setting to change when we deploy to all users. (iii) create two groups in users, each group only see a group of new items. But this negatively affect user's experience.

In all, for author side metrics, we should think how traffic is taken away between experimental and control groups, new items and old items. Clearly, splitting new items, users or even old items into different groups may change the setup between AB test and full deployment. How to mitigate this difference? If we simutaneously split items and users, the content pool can be greatly reduced and decrease user experience.


## Other tricks

### Important metrics and their relations

DAU and retention are the core metrics. Most used metrics today for retention are LT7 and LT30. Here LT-k means after user login in today, the number of days user log in in the next k days (including today).

LT can be tricky: say a strategy drives away a lot of light users. LT will increase, but DAU decrease. 

Total num of impression, clicking may not be highly correlated with the usage time. Say we recommend long content to users, they will spend more time on reading, but the amount of items they read will decrease. The usage time affect DAU and retention, while num of clicks affect ads monatization. 

For user generated content (UGC) platforms, release volume and penetration rate are also important. 

### Retrieval

1. The total amount of retrieval should be fixed. More retrieved items provided, higher the metrics will be, but there will be also more computation in the light ranker.
2. Two tower and item to item are the most important retrieval models that take most of the quota.
3. Some less popular models get less quota. Fix the total amount, adding some retrieval models can imporve the metrics.
4. Bucketize content pools, by time, by consumers, etc.
5. One model can be applied to multiple content pools.

#### Two tower

Directions. 
1. negative samples. 
2. NN structure. 

 Multi-vector, where the user tower output multiple embeddings representing different perspective of the user. The reason of not having this in item tower is, we don't want to maintain many ANN index.

3. Training, use batch negative,  self-supervised

#### item to item (I2I)

1. ItemCF and varients (Swing, and online models)
2. User two tower or GNN to compute the embedding, use that to measure the similarity of items

#### Other retrieval models

1. UserCF
2. User to author to item, user to author to author to item
3. Novel methods like PDN, Deep retrieval, SINE, M2GRL

### Ranking

#### Heavy ranker

1. the foundation of the model takes discrete features and generate embeddings. Then it sends the embeddings and continuous features to MLP. Widening and deepening the MLP can improve the model capability. Using bilinear or LHUC to cross feature can also improve the model. Finally you can also do feature engineering.
2. The upper level of the model predicts multiple metrics, such as like rate, click rate... One can add new metrics to improve. Use MMoE or PLE structure may improve the perf. Correct the position bias may also work.

#### Light ranker

1. Can use two tower or three tower models here.
2. Can distill the heavy ranker, by pointwise, pairwise or listwise. However, if there is bug in heavy ranker, the light ranker can be polluted without easy notice.

#### User behavior sequence modeling

1.  Baseline, take average of the recent-interacted item embeddings. More advanced models like DIN and SIM.
2. Add sequence length to look at more items. This is restricted by infra
3. Categorize items and filter by the new item category
4. Use features more than ID to generate the embedding

#### Online learning

Model is refreshed every hour / minute by additive training, and refreshed completely everyday.

The resource used for online learning can be impacted by the number of AB tests. Each heavy ranking model being tested will consume a set of computation resource. In addition we have a set for holdout and a set for fully deployed model. 

While online learning is very important for metric improvement, its computation requirement greatly restrict the iteration of models.

#### Old model

Old model have been trained on a lot of data, thus often outperform new models even there is known issues. This cause some problems:

1. How to evaluate new model. - use the same data to train models from scratch.

2. How to make new model catch up. - Reuse embedding layer as much as possible. - Use old model to distill. Say the true interaction is $y$, old model's prediction is $p$, train the new model to predict $y+p\over 2$.

### Diversity

#### Heavy ranker



### Special user groups

### Different interactions