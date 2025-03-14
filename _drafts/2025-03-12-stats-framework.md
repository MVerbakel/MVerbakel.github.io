---
layout: post
title: "Analytics framework"
mathjax: true
---

In this post, I want to talk about some standard approaches for different analytics problems. The goal is for this to be a bit of a one-stop reference, so it won't go into detail on the actual methods, but I'll try link any relevant sources.

---
# Statistical tests

One possible decision tree for selecting an appropriate statistical test depending on measurement and intent:

<img src="/assets/stat_tests.png" alt="tests" width="100%"/>
Source: [Statistical Rethinking, figure 1.1](https://www.taylorfrancis.com/books/mono/10.1201/9780429029608/statistical-rethinking-richard-mcelreath)

The author refers to these as "statistical golems", noting they have no wisdom about whether the context is appropriate, they just perform a procedure. Without our guidance and skepticism, they may do nothing useful. It's important to understand how they work in order to interpret the results.

---
# Comparisons

## Comparing variables (associations)

numerical vs numerical
categorical vs categorical
numerical vs categorical
binary

Correlation
Covariance

## Comparing data (distributions)

I have two samples, are they the same?

---



## Martinde's method

Use earlier descriptive stats to quantify how much of an outcome is explained by them changing.