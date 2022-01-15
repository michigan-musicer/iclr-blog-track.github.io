---
layout: post
title: A gentle overview of Contrastive Learning with Hard Negative Samples
authors: Anonymous
tags: [contrastive learning, sampling, introductory]  # This should be the relevant areas related to your blog post
---

# A gentle overview of Contrastive Learning with Hard Negative Samples
### Table of contents
- Introduction
- Background on contrastive learning
- Paper motivation
- Methods
- Evaluation
- Constructive criticism
- Conclusion
- References

## Introduction
Contrastive learning techniques have existed since the early days of deep learning. Recently, researchers have been able to use them for significant performance gains in unsupervised and self-supervised settings in computer vision, natural language processing, and other applied fields of machine learning. However, there is still little in the way of introductory material on contrastive learning techniques. Contrastive learning has not been around long enough to become a standard part of university coursework, and most of the people exploring and using contrastive learning are experienced machine learning practitioners who have the theoretical and practical experience to navigate through literature. 

This blog post walks through Contrastive Learning with Hard Negative Samples from ICLR 2021. The analysis is written at a level intended for an advanced undergraduate student or beginning graduate student. This post assumes basic familiarity with machine learning, but little to no prior knowledge on contrastive learning. The main purpose of this post is to expose a newer machine learning practitioner to contrastive learning -- not excessively holding back on meaningful content, but also not overwhelming the reader with too many new ideas at once.

## Background on contrastive learning
Contrastive learning is a method to learn representations (also known as embeddings) of datapoints such that two "similar” datapoints have similar representations and two “dissimilar” datapoints have dissimilar representations. The end result of learning contrastive representations is an embedding function that returns a representaiton of a given datapoint in the learned embedding space. It’s typical for an iteration of contrastive learning to consider an *anchor* datapoint (the current datapoint to compare to) and then decrease the distance to *positive* (similar) datapoints and increase the distance to *negative* (dissimilar) datapoints.

Traditionally, contrastive learning is treated as a self-supervised problem: a flavor of unsupervised learning where labels are assigned without outside information to help categorize information for learning. There are other ways to use contrastive learning (in particular, supervised methods) that we will not get to in this post.
#### A common (misleading) analogy for contrastive learning
A common way to explain contrastive learning is to essentially recast the process as a classification problem. This post will use this analogy and then explain it is a bad analogy.

Consider learning contrastive representations on a dataset of cats and dogs. One could imagine that cats should be similar to cats and dogs should be similar to dogs. Using this logic, consider a batch of cats and dogs from our dataset. For each iteration through the batch, if the current anchor is a cat, then we decrease the distance of the anchor's representation to representations of other cats in the batch, and we increase the increase the distance of the anchor's representation to representations of dogs in the batch. Vice versa if the anchor is a dog.
~~Pictures here~~
~~. Then you can adjust the embedding space so that the negative is further from the anchor and the positive is closer to the anchor. ~~

Many papers – including the one analyzed in this post – use this misleading example or something similar to it because it’s easy to think about and it makes sense. It has its merits as an explanatory tool, but contrastive learning does not have the same sense of class hierarchy as a human observer. Without providing labels, it's not possible to guarantee that a contrastive learning framework will learn the "correct" embedding space on its own.
It is more accurate to think of datapoints as just that -- points in a space. Using analogies like this assigns more ability and agency to contrastive learning frameworks than what's really there.

#### Contrastive learning frameworks
Contrastive learning owes much of its recent success to the same innovations that sparked the deep learning revolution: big data and greater computational resources. Modern contrastive learning frameworks rely on these things 

The first version of MoCo (Momemtum Contrast for Unsupervised Visual Learning) came out in November 2019. MoCo maintains a directory of raw images and distorted images. For a given anchor image, its distortion is a positive, and all other images in the dictionary are negatives.

The first version of SimCLR (Simple Contrastive Learning Representations) came out slightly later than MoCo, in February 2020. It was built on a simple premise: in each batch, run two image augmentations on a given anchor image and treat the augmented result as a positive, then treat the rest of the batch as negatives.

With regards to unsupervised pretraining, MoCo and SimCLR both significantly outperformed preexisting methods. Additionally, the representations learned by both frameworks end up being useful for downstream tasks (that is to say, using the contrastive learning representations instead of the raw data often boosts model performance on that model's own task). Because they closed the gap so much, both of the papers for these frameworks have thousands of citations and are widely credited for sparking a surge in research into and application of contrastive learning. The frameworks themselves are not the focus of this post, but any discussion of contrastive learning and its impact in the last few years will include MoCo and SimCLR. 

#### Contrastive learning is not the only way forward
There are a number of semi-supervised and unsupervised techinques other than contrastive learning gaining traction in the machine learning community today. Bootstrap Your Own Latents (BYOL) beats traditional contrastive learning methods on  by employing a similarity measure instead of a contrastive measure including positives and negatives. Additionally, late in 2021 within the field of computer vision, masked autoencoder frameworks (MAE) caused a stir for showing higher performance than supervised frameworks on certain tasks. Contrastive learning is dramatically more powerful for representation learning than more traditional unsupervised techniques like clusteirng or PCA, but it is certainly not the only option out there.
## Paper motivation
Contrastive Learning with Hard Negative Samples proposes a new contrastive learning method that focuses on sampling *hard negatives* – that is, the negative examples that are closest to the anchor in the embedding space. The primary benefit of this is faster convergence, as the harder negatives give more information than negatives already far away from the anchor. Intuitively, knowing that an anchor $P$ and a nearby negative $N_1$ should be far apart is more informative than knowing $P$ and a faraway negative $N_2$ should be far apart, because $P$ and $N_2$ are already far apart.

#### Current literature does not focus much on finding "good" negatives
In the contrastive learning literature, negatives usually don't receive much attention. MoCo and SimCLR simply assume that all other examples in a batch not related to the anchor (i.e. not a noised version of the anchor, not an augmentation of the anchor) are negative, and the frameworks are still very effective with this assumption. 

Partially due to the success of these frameworks and partially due to relatively limited experiments with known ways to sample negatives, it's possible to argue that searching for negative pairs doesn’t raise model performance enough to be worth spending time on. This paper points to work in data mining and metric learning to suggest otherwise. Negative sampling in data mining and metric learning has been shown to reduce training times because the harder the negative is, the more quickly an algorithm separates the representations of different types of datapoints from each other. Contrastive learning, the paper claims, is another unsupervised process which may benefit from the same idea.

#### Relevant prior work: NCE loss and debiased contrastive learning

This paper states it is built on a contrastive loss called noise-contrastive estimation (NCE). However, the loss function they refer to as NCE is

$$\mathbb{E}_{x\sim p, x^+ \sim p^+_x, \{x^-_i\}^{N}_{i=1} \sim q } \bigg[ -\log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + \frac{Q}{N}\sum_{i=1}^N e^{f(x)^Tf(x^-_i)}} \bigg]$$

which does not match any formulation of NCE loss that I can find. This instead seems to be an "ideal" loss function used to motivate [], which focuses on learning a *debiased* contrastive learning loss function.

$$\mathbb{E}_{x, x^+, \{x^-_i\}^{N}_{i=1} } \bigg[ -\log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + \frac{Q}{N}\sum_{i=1}^N e^{f(x)^Tf(x^-_i)}} \bigg]$$

Discussing NCE is still informative because it is a less complicated introduction to the ideas behind this paper's loss function formulation. 

The original NCE paper formulates NCE loss as 

$$-\frac{1}{2T} \sum_t \ln [h(x_t;\theta)] + \ln[1 - h(y_t; \theta)]$$

with the following definitions:
- $T$ is total number of examples in dataset
- $x_t$ is the $t$th observation in dataset with distribution $p_m$
- $y_t$ is the $t$th observation of a *noise* dataset with distribution $p_n$
- $\theta$ is a parameter vector that describes $p_m$
- $h(u;\theta) = \frac{1}{1 + e^{-G(u;\theta)}}$, where 
	- $G(u; \theta) = \ln p_m (u;\theta) - \ln p_n(u)$

At a high level, NCE is essentially logistic regression to estimate parameters for probability distributions. NCE loss is minimized when the datapoints $x_t$ are very likely to come from a distribution described by $\theta$ and the noise points $y_t$ are very unlikely to come from that same distribution. Notice that how those distributions are described by $\theta$ isn't actually necessary -- $\theta$ could be parameters for any distribution we want.

NCE is most famous for being a formalization of a component of the framework used to learn Word2Vec embeddings. In contrastive learning, it’s the simplest loss function that offers a view of contrastive learning from the perspective of optimizing probability distributions. 

Debiased contrastive learning, which is the method this paper actually seems to be pointing at, is named so because its goal is to reduce the impact of false negatives on contrastive learning representation efficacy. ("Debiased" shows up similarly in machine learning theory literature.) Debiased contrastive learning is motivated by the idea that false negatives are common and decrease model accuracy, and that one can adjust the loss function to account for these false negatives and maintain model performance.
Contrastive learning with hard negatives ends up splitting off from the same motivating "ideal" loss as debiased contrastive learning. For brevity and to avoid inundating the reader with symbols, I will not go in depth on the latter.

## Methods

Our goal is to arrive at 

$$\mathbb{E}_{x \sim p, x^+ \sim p^+_x} \bigg[ - \log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + \frac{Q}{T^-} (\mathbb{E}_{x^- \sim q_{\beta}} [e^{f(x)^Tf(x^-)}] - \mathbb{E}_{x^- \sim q_{\beta}^+} [e^{f(x)^Tf(v)}])} \bigg]$$

At a high level, this objective is minimized when one minimizes the expected distance between anchors and their positives and maximizes the expected distance between anchors and their negative, **while also prioritizing sampling hard negatives**. This is quite messy and convoluted, so we'll get back to this.

We start with the same motivating objective for debiased contrastive learning, 
$$\mathbb{E}_{x \sim p, x^+ \sim p^+_x, \{x_i^-\}}^N_{i=1} \sim q \bigg[ - \log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + \frac{Q}{N}\sum_{i=1}^N e^{f(x)^Tf(x_i^{-})}} \bigg]$$

Let's break this down:
- $p$ is the sample space, i.e. the distribution that all datapoints are drawn from
- $f(x)^Tf(x^+)$ and $f(x)^Tf(x^-)$ are inner products of anchor and corresponding positive / negative, which represents the distance between them in the embedding space
- $Q$ is a hyperparameter (usually set to $N$) and $N$ is the number of negatives for anchor $x$

This value will be maximized when $e^{f(x)^Tf(x^+)}$ is maximized and $e^{f(x)^Tf(x_i^{-})}$ is minimized for a corresponding positive $x^+$ and all the corresponding negatives $x^-$. In other words, we want to minimize the distance between anchors and positives, and we want to maximize the distance between anchors and negatives.

To prioritze sampling hard negatives, we want to consider what distribution we're getting our negatives from. $q$ is usually just set to $p$ -- meaning we just draw negatives uniformly from the data distribution -- but if we could somehow adjust that to draw datapoints further away from the anchor, we could get harder negatives.

The paper proposes the following distribution instead:

$$q_\beta^- (x^-) := q_\beta (x^- | h(x) \neq h(x^-))$$
where $$q_\beta(x^-) \propto e^{\beta f(x)^Tf(x^-)} \cdot p(x^-), \beta \geq 0$$

Breaking this down:
- $h(x)$ and $h(x^-)$ are the latent classes of the anchor and the corresponding negative -- which is to say, the actual classes of those datapoints
	- Note that in this distribution, we are pretending we know the actual classes of these datapoints and that we can get all the negatives that are of different classes from the anchor -- that is, true negatives only.
- $\beta$ is a hyperparameter that controls how important it is to sample close to the anchor. The higher the $\beta$ value, the more importance that is placed on sampling close to the anchor.

We can't actually use this distribution because we don't know these latent classes, and even if we could, it turns out that it's not clear how to sample efficiently from this distribution. To deal with this, we condition on $h(x) = h(x^-)$ and split $q_\beta(x^-)$ into

$q_\beta(x^-) = \tau^- q_\beta^-(x^-) + \tau^+q^+_\beta(x^-)$
where $q^+_\beta(x^-) = q_\beta(x^- | h(x) = h(x^-)) \propto e^{\beta f(x)^Tf(x^-)} \cdot p^+(x^-)$

$$q^+$$ handles the case where the latent classes of $x, x^-$ are the same (false negative). $\tau^-$ and $\tau^+$ are temperature variables, and I'm honestly still confused what these do. Machine learning practitioners seem to agree that these temperature variables are necessary to "restrict" the embedding space and return a good solution, but there's not a good explanation I can find of why this is the case.

Vague hyperparameters aside, the paper next seeks to isolate $q_\beta^-(x^-)$, which is essentially the probability of drawing a true positive. We can rewrite $q_\beta^-(x^-)$ as 

$$\big( q_\beta^-(x^-) - \tau^+q^+_\beta(x^-)\big) / \tau^-$$

Now we revisit the starting equation:

$$\mathbb{E}_{x \sim p, x^+ \sim p^+_x, \{x_i^-\}}^N_{i=1} \sim q \bigg[ - \log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + \frac{Q}{N}\sum_{i=1}^N e^{f(x)^Tf(x_i^{-})}} \bigg]$$

If we fix Q and take the limit as $N \to \infty$, we get

$$\mathbb{L}(f,q) = \mathbb{E}_{x \sim p, x^+ \sim p^+_x \bigg[ - \log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + Q\mathbb{E}_{x^- \sim q} e^{f(x)^Tf(x_i^{-})}} \bigg]$$

This lets us recast the first equation as an approximation of the second with $N$ negatives. We can now rewrite the expectation in the denominator $\mathbb{E}_{x^- \sim q} e^{f(x)^Tf(x_i^{-})}$ to arrive at the desired objective function:

$$\mathbb{E}_{x \sim p, x^+ \sim p^+_x} \bigg[ - \log \frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)} + \frac{Q}{T^-} (\mathbb{E}_{x^- \sim q_{\beta}} [e^{f(x)^Tf(x^-)}] - \mathbb{E}_{x^- \sim q_{\beta}^+} [e^{f(x)^Tf(v)}])} \bigg]$$

The authors proceed to use Monte Carlo sampling to approximate $$p$$ and $$p^+$$. This step is just to move the method from statistical theory into something computationally feasible, so for brevity I will leave it out.

## Evaluation
#### Theoretical results

The paper draws two theoretical conclusions about the formulated objective.

The first conclusion is that “hard sampling interpolates between marginal and worst-case negatives”. Essentially what this means is that by controlling $\beta$, one can always get either the least hard (furthest) negatives or hardest (closest) negatives. This is necessary to establish but not too surprising.

The second conclusion is twofold: "optimal embeddings $f^{\*}$ on the hypersphere for worst-case negative samples “almost surely” guarantee that $f^{\*}(x) = f^{\*}(x^+)$", and letting $v_c = f^{∗}(x)$ for any $x$ such that $h(x) = c$ (so $v_c$ is well defined up to a set of $x$ of measure zero), $f^{∗}$ is characterized as being any solution to the [Tammes ball-packing problem]".

The first half of this states that if you pick very hard negatives, then anchor points and their positives will be almost equal, matching one of the goals of contrastive learning. The second half of this states that anchors and their negatives will be separated about as far away as mathematically possible. As we’ll see, in practice too high of a $\beta$ value will cause performance to deteriorate as the algorithm starts sampling lots of false negatives. 
The second half relies on viewing the minimum-cost embedding function $f^{*}$ as a solution to the Tammes ball packing problem. Curious readers may do their own reading on it, but a solution to the Tammes problem essentially maximizes the distance between circles on a sphere (which can be thought of as corresponding to maximizing the difference between similar datapoints in the embedding space). The proof is highly mathematical, and it's not necessary to understand the inner workings of the proof to understand the paper at a high level.


#### Empirical results
Older contrastive learning papers that were published before MoCo and SimCLR are usually more theory-focused and don’t often include empirical results. Nowadays, due to contrastive learning's proven efficacy, there is more expectation for meaningful empirical results. This is one of the main points of critique I have for the paper, which I will discuss when going through constructive criticism.

The paper shows results of using contrastive learning with hard negatives on classification tasks using image representations, graph representations, and sentence embeddings (computer vision, graph learning, and natural language processing fields respectively). Positive results were shown on all three, but the paper focuses most of its analysis on the image classification tasks -- as will I.

The new loss function proposed in this paper outperforms both the base SimCLR loss function (from the original 2020 paper) as well as the debiasing loss function explained earlier. However, too much of a good thing can be bad, and when $\beta$ is set to too high of a value, performance suffers. (The paper's results show a peak in performance at $\beta = 0.5$.)

Negatives end up being more dissimilar to the anchor than SimCLR -- which is desired -- but positives are less similar. The paper shows that the overlap between positive and negative histograms is reduced, which implies that it is easier to distinguish between the two for a given anchor point than with SimCLR. So overall, this aspect of the framework's performance is improved. 

## Constructive criticism
#### Misleading expalantory example
As mentioned before, this paper falls into the same pitfall that a lot of other papers fall into and uses a motivating example implying classification with a specified class hierarchy. Again, contrastive learning isn't really capable of this.
#### SimCLR test not up to date
The paper uses the original SimCLR framework; there is now a SimCLRv2 with improved performance over the original, and it would be more fair to compare performance to that updated version. Using the old version is understandable because SimCLRv2 was published relatively close to the ICLR 2021 deadline. However, it would improve the paper's credibility to have results on the new SimCLRv2.  
#### Weak computer vision tests
From the perspective of a computer vision practitioner, the datasets used to evaluate the novel loss function’s performance on computer vision tasks is too small, and there is no testing of the contrastive embeddings on complex downstream tasks. (Nowadays, image classification is often too simple to serve as a good litmus test by itself.) Seeing as to how contrastive learning is most prominently used in computer vision settings, this reduces the practical credibility of the results by a fair amount. The paper would have higher credibility if it had run more rigorous tests on more difficult downstream tasks, such as image segmentation or object detection, to better establish the legitimacy of their results. 
#### Very little focus on time to convergence
From a practical standpoint, neither the paper’s theoretical or empirical results focus that much on the reduced training time. This was one of the main motivations for the paper, so it is odd that so little analysis is dedicated to it. Without a justification in regards to time spent training or number of iterations to converge, there is little convincing reason to adopt it over simpler existing contrastive loss functions.

## Conclusion
Contrastive Learning with Hard Negative Samples continues the exploration of negative-focused contrastive learning methods. The paper is relatively easy to understand and communicate, and it does a good job of connecting related literature on this specific subject for less familiar audiences. I do believe the empirical results are shallow for application in computer vision. Additionally, I think there's not enough focus on an important question -- whether hard negative sampling in contrastive learning mimics hard negative sampling in other unsupervised tasks and allows for faster convergence. However, the results are theoretically robust and clearly add something new to the contrastive learning toolkit, and the code implementation is smooth. Overall, this is a strong paper worthy of its ICLR 2021 acceptance.
