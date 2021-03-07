---
layout: post
title: "A/B testing: an overview"
mathjax: true
---


Part book review for trustworthy experiments, part general notes on running AB tests.

Type S: sign error (probability estimate being in the wrong direction)
Type M: magnitude error (factor by which magnitude might be overestimated).

Multiple testing: separate metrics into 3 groups with descending p-value limits based on priority (primary can be 0.05, secondary 0.01, but third order should 0.001). How much do you believe the null is true pre experiment? Stronger the belief, the lower the significance level you should use.

Internal/external validity - generalising results.

Outliers
Skew
Capping/winsorizing
Variance reduction: regression adjustment, cuped
Robust SE
Clustered SE

