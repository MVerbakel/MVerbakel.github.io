---
layout: post
title: "Causal inference"
mathjax: true
---

Summary of key methods, pros and cons, basic code/implementations.


The fundamental problem of causal inference is that we can’t observe both outcomes for the same customers. So, if a customer received the discount, we will never know what value or retention he would have had without a coupon. It makes causal inference tricky.

That’s why we need to introduce another concept — potential outcomes. The outcome that happened is usually called factual, and the one that didn’t is counterfactual. We will use the following notation for it.



Methods:

Linear regression

Matching

Propensity score

IV

Difference in differences

Synthetic controls



Experimentation

