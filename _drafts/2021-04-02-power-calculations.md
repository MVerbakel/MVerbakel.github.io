---
layout: post
title: "Statistical power: concept and calculations"
mathjax: true
---

A power analysis is done before running an experiment to determine the sample size required for a target statistical power, significance level, and minimum detectable effect. 
The power is the chance of detecting an effect when one actually exists (or the inverse of the type II error rate, false negatives as a proportion of those where Ho false). By holding the other factors constant, we can determine how large the sample needs to be to achieve given settings (noting the standard error decreases as the sample increases). This is an important step, because if you're test is underpowered (e.g. very low value like \<50%), it's unlikely you will be able to detect an effect even if it exists. By simulating different settings, the power analysis helps us trade-off the different components depending on how much we care about type I (False Positive) and II errors (False Negative).

## Inputs:

- Significance level (lower significance requires larger sample), typically set at 0.05 (5% false positive rate);
- Target power (higher power requires larger sample), typically set at 80% but should depend on how much you care about type II errors.
- Minimum detectable effect (smaller effects require a larger sample to detect), which should be based on what is a meaningful effect for your use case.

You can of course flip the formula and calculate the power given the sample size you're able to achieve (or equivelantly, the acceptable runtime). 

## Plots



## Notes:
- The power analysis is typically done for the primary metrics of interest. However, if you want to ensure you also have sufficient power to detect effects in supporting metrics, or different segments, then you can also run the same analysis for these (using the maximum sample size as the final number).
- Due to the high variance of many continuous metrics (e.g. amount spent is typically skewed), binary metrics like conversion typically require a smaller sample size. However, this shouldn't be the only reason for using a binary metric as the primary. It first of all should align with your hypothesis.

Bonferroni correction: alpha/k for k hypotheses