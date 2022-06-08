# The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective

### Abstract

As various post hoc explanation methods are increasingly being leveraged to explain complex models in
high-stakes settings, it becomes critical to develop a deeper understanding of if and when the explanations
output by these methods disagree with each other, and how such disagreements are resolved in practice.
However, there is little to no research that provides answers to these critical questions. In this work,
we introduce and study the disagreement problem in explainable machine learning. More specifically,
we formalize the notion of disagreement between explanations, analyze how often such disagreements
occur in practice, and how do practitioners resolve these disagreements. To this end, we first conduct
interviews with data scientists to understand what constitutes disagreement between explanations (feature
attributions) generated by different methods for the same model prediction, and introduce a novel
quantitative framework to formalize this understanding. We then leverage this framework to carry out a
rigorous empirical analysis with four real-world datasets, six state-of-the-art post hoc explanation methods,
and eight different predictive models, to measure the extent of disagreement between the explanations
generated by various popular post hoc explanation methods. In addition, we carry out an online user
study with data scientists to understand how they resolve the aforementioned disagreements. Our results
indicate that state-of-the-art explanation methods often disagree in terms of the explanations they output.
Worse yet, there do not seem to be any principled, well-established approaches that machine learning
practitioners employ to resolve these disagreements, which in turn implies that they may be relying on
misleading explanations to make critical decisions such as which models to deploy in the real world. Our
findings underscore the importance of developing principled evaluation metrics that enable practitioners
to effectively compare explanations.

### Repository

This repository contains code for the empirical analysis performed in the paper ["The Disagreement Problem in Explainable Machine Learning"](https://arxiv.org/pdf/2202.01602.pdf). To measure the extent of disagreement between the explanations generated by various popular post hoc explanation methods, we leverage the evaluation metrics and framework proposed in the paper to perform an empirical analysis with four real-world datasets, six state-of-the-art post hoc explanation methods, and eight different predictive models.
