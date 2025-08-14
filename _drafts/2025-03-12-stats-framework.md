---
layout: post
title: "Analytics framework"
mathjax: true
---

In this post, I want to talk about some standard approaches for different analytics problems. The goal is for this to be a bit of a one-stop reference, so it won't go into a lot of detail on the methods, but I'll try to link any relevant sources.

# Statistical tests

One possible decision tree for selecting an appropriate statistical test depending on measurement and intent:

<img src="/assets/stat_tests.png" alt="tests" width="90%"/>
Source: [Statistical Rethinking, figure 1.1](https://www.taylorfrancis.com/books/mono/10.1201/9780429029608/statistical-rethinking-richard-mcelreath)

The author refers to these as "statistical golems", noting they have no wisdom about whether the context is appropriate, they just perform a procedure. Without our guidance and skepticism, they may do nothing useful. It's important to understand how they work in order to interpret the results. Nonetheless, the visual is a useful reminder of which test to use for what. Below I'll break these down with more specific examples.

### Which t-test

Paired (same test subject measuring before and after) vs unpaired (two groups) t tests. The unpaired can either assume the same variance or unequal variance (more conservative). Then we can either do one sided (test one direction, e.g. larger) or a two sided (test both directions of difference) test. Generally use the two sided (don't assume direction known) and unequal variance as more conservative.

## Differences

One common scenario is testing whether two samples or groups are significantly different (e.g. different variants in an experiment, or different groups from observational data). 

First we must define what we mean by different:
- Difference in the mean/median, difference in the variability (variance), difference in the shape (skewness/kurtosis), or even difference in the associations?
- Direction of difference: one/two sided, non inferior
- Number of dimensions: one variable or multiple



Here's how you might approach this:

Distributions (e.g., I have 2 samples, are they the same?):
- T-test (Independent): Compares the means of two independent samples assuming normal distribution.
- Mann-Whitney U test: A non-parametric test comparing the distributions of two independent samples when you cannot assume normality.
- Kolmogorov-Smirnov Test: Compares the cumulative distributions of two samples, useful when you don’t assume normality.
- Paired T-test: Used when the samples are not independent (i.e., when you are measuring the same group before and after an intervention).

## Relationships (associations)

numerical vs numerical
categorical vs categorical
numerical vs categorical
binary vs binary
numerical vs binary
categorical vs binary

Correlation
Covariance

---
## Frameworks

Change in metric
Intervention at time x

--- 
## OTHER

When to use simulation vs experiment vs observational analysis....

Link between t tests and ANOVA and regression


Lowess and Loess

# Tips

Think of simple and extreme examples, how would they change things.


