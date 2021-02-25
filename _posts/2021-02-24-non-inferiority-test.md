---
layout: post
title: "Non-inferiority testing: t-test with a margin"
mathjax: true
---

Non-inferiority (or superiority) tests are just one-sided tests with a margin, but they're pretty useful in experimentation. For example, let's say you're adding a new feature to your web shop. You're primarily interested in increasing revenue, but at the same time you don't want to hurt other metrics like the average page load time, or volume of customer service queries. Metrics that you just want to keep an eye on like this are sometimes called 'guardrails'. For these we can use a one-sided test with a margin to check if the difference exceeds some acceptable threshold. If we conclude the difference exceeds the threshold, the new change is inferior to the current, and if not, it's non-inferior.

Like with all hypothesis tests, it's impossible to prove there is no effect. Instead, we test whether the difference is not worse than some meaningful threshold. For example, we may be ok with some increase in customer service volumes (particularly if it may be temporary), but decide anything above 5% is too high a cost. In this case 5% will be our threshold, and we'll test whether the change exceeds this or not. Depending on the direction of 'good', this gives us the following hypotheses (using a t-test on two independent samples as an example):

**If high values are good:**

In this case rejecting the null hypothesis implies that the mean is greater than the base group by at least the threshold (for superiority).

- Ho: mean2 <= mean1 + threshold or equivelantly mean2-mean1 <= threshold
- Ha: mean 2 > mean1 + threshold or equivelantly mean2-mean1 > threshold

**If high values are bad:**

In this case rejecting the null hypothesis implies that the mean is less than the base group by at least the threshold.

- Ho: mean2 >= mean1 - threshold or equivelantly mean2-mean1 >= -threshold
- Ha: mean2 < mean1 - threshold or equivelantly mean2-mean1 < -threshold

Note, for experiments mean 1 would be the base or control group, and mean2 would be the treatment group.

### Python function:

Note, we shift the base mean to allow some gap (subtracting the threshold), then it's just a standard one-sided t-test comparing it to the second group mean. 

```python
from scipy.stats import ttest_ind_from_stats
import numpy as np

def non_inferiority_ttest(mean1, stddev1, n1, mean2, stddev2, n2, relative_difference, equal_variance=False, increase_good=True):
    '''
    Perform a one-sided t-test with a non-inferiority threshold for two independent samples.
    mean1: group 1 mean
    stddev1: standard deviation of group 1
    n1: number of observations in group 1
    mean2: group 2 mean
    stddev2: standard deviation of group 2
    n2: number of observations in group 2
    relative_difference: threshold as a percentage of the base group statistic (e.g. 0.1=10% difference)
    equal_variance: if True, Welch's t-test
    increase_good: if True, Ho: mean2 <= mean1 + threshold.
    '''
    
    delta = relative_difference * mean1

    if increase_good:
        threshold = mean1 - delta
    else:
        threshold = mean1 + delta

    tstat, pval = ttest_ind_from_stats(mean1=threshold, 
                                       std1=stddev1, 
                                       nobs1=n1, 
                                       mean2=mean2, 
                                       std2=stddev2, 
                                       nobs2=n2, 
                                       equal_var=False)
    
    return tstat, pval

```

### Example: Increase is good, threshold is 0.1 (10%). 

Ho: mean2 >= mean1 - mean1*0.1 

```python
np.random.seed(26)
group1 = np.random.normal(47, 3, 100)
group2 = np.random.normal(42, 3.5, 105)
relative_difference_threshold = 0.1
mean_group1 = np.mean(group1)
mean_group2 = np.mean(group2)
stddev_group1 = np.std(group1, ddof=1)
stddev_group2 = np.std(group2, ddof=1)

tstat, pval = non_inferiority_ttest(mean1=mean_group1,
                                    stddev1=stddev_group1, 
                                    n1=len(group1), 
                                    mean2=mean_group2, 
                                    stddev2=stddev_group2, 
                                    n2=len(group2), 
                                    relative_difference=0.1, 
                                    equal_variance=False, 
                                    increase_good=True)

print('One sided ttest: t value = {:.4f}, pval = {:.4f}'.format(tstat, pval/2.0))
```
```python
One sided ttest: t value = 1.3677, pval = 0.0865
```

### Notes
- You could also do a two-sided test and estimate the effect (checking if the estimate is above or below the threshold), but this will require a larger sample. 
- I've shown the non-inferiority approach for a two sample ttest, but the concept works equally well for other metrics and tests (e.g. proportions, or the mann-whitney rank test).
- If you have trouble deciding the threshold, consider what size change would you celebrate if it was the primary metric for the experiment? Or perhaps, what is the cost of hurting this metric, and when would that cost be too high?