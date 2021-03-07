---
layout: post
title: "Hypothesis testing: Two sample tests for continuous metrics"
mathjax: true
---

This post is a quick reference on some of the statistical tests I use most often for comparing a continuous metric across two independent samples. Note, in my work the samples are relatively large, so I don't focus much on issues specific to small samples.

## Motivating example

 Let's say we run an e-commerce site. We think that adding a short description to each product on the main page will increase the probability of a visitor purchasing, thereby increasing our total profit. So we run an AB experiment, taking two independent random samples from the population and showing one group the current site and the other the new site (with descriptions). The observed difference in the average profit per visitor is our estimated average treatment effect (i.e. how much we expect profit to change on average per visitor with the new site). 
 
 However, as we're taking samples, we expect some variation due to sampling. If we repeated the experiment many times, the observed difference will vary around the true population difference (creating the sampling distribution). Therefore, when making our decision on the outcome, we also need to take into account the uncertainty of the estimate. We do this by considering the size of the effect, relative to the variation in the data. The more variation in the outcome variable, the larger the difference needs to be for us to be confident it's unlikely just random variation due to sampling. 

 Below I'll formalise this process using the hypothesis testing framework.

## Hypothesis testing steps

**Step 1: Hypothesis** <br>

Two-sided test (checking for any difference, larger or smaller):

- Ho: mu1-mu2=0, there is no difference in the means of group 1 and 2 <br>
- Ha: mu1!=mu2, there is some difference (not equal)

Alternatively, a one-sided test would look something like:

- Ho:mu1-mu2 < threshold, or Ho:mu1-mu2 > threshold.

**Step 2: Significance** <br>

The level of significance (alpha), is the acceptable false positive rate. A common choice is 0.05, which means we expect our test to incorrectly reject the null hypothesis 1/20 times on average (i.e. we conclude there's sufficient evidence the means are not equal, when there's actually no difference). The other 95% of the time, we expect to correctly accept the null. The smaller the alpha selected, the larger the sample needed. It's common to see a larger alpha such as 0.1 in business settings (where more risk is tolerated), and a smaller alpha such as 0.01 in medical settings. Note, there is no significance level that is correct 100% of the time, we always have to trade-off the probability of a false positive, with having enough power to detect an effect when it exists.

If it's a two sided test, we test whether the difference is positive or negative (both directions). In this case we divide alpha by 2 (e.g. for p=0.05, we use 0.025 on each side), and the result is significant if it's in the top or bottom 2.5%. For a one-tailed test we put the full 0.05 in one tail and so the result is significant if it's in that 5%.

**Step 3: Test statistic** <br>

Below we will discuss the specifics, but this is essentially calculating the standardised difference between the two groups (relative to the variation in the data). We can then look at the sampling distribution under the null to determine how common a value like this (or larger) is, when the null hypothesis is true.

**Step 4: p-value** <br>

 The p-value measures the area under the curve for all values above the test statistic (i.e. for results as large, or larger than this). If it's very uncommon when the null is true (i.e. less than alpha), we reject the null hypothesis. In this case we conclude it's unlikely that there is no difference given the data. In the inverse case, p > alpha, we accept the null (we don't have enough evidence to indicate it's not true given the data, i.e. we commonly see differences this large by chance when the null is true). Note, this doesn't mean we have proven the null is true. Either the null is true, or the effect was too small for our test to detect. 

Some confuse the p-value as relating to the probability the null or alternate are true, but remember it's conditioned on the null hypothesis being true:

$$ p value = P(\text{observed difference} \:|\: H_\text{o} true)$$

<br>

## Tests for comparing means (independent samples):

### t-test

The t-test was developed by a scientist at Guinness to compare means from small samples. It's similar to the z-test, but compares the test statistic to the t-distribution instead of the normal distribution, so it has thicker tails. However, it converges to normal as the sample size increases, so it's commonly used even when the sample is large. Below we calculate the t value (test statistic):

$$ t = \frac{\bar{x_1} - \bar{x_2}}{s_p \sqrt{(\frac{1}{n1} + \frac{1}{n2})}}$$

$$s_p = \sqrt{\frac{(n_1 - 1)s_\text{x1}^2 + (n_2 - 1)s_\text{x2}^2}{n_1 + n_2 - 2}}$$

Where:
- $$\bar{x_1}$$, $$\bar{x_2}$$ are the sample means,
- n1,n2 are the sample sizes 
- $$s_\text{x1}^2$$, $$s_\text{x2}^2$$ are the sample variances
- $$s_p$$ is the pooled sample variance (estimate for population variance $$\sigma^2$$)
- $$n_i − 1$$ in the formula is the degrees of freedom for each group. The total degrees of freedom is the sample size minus two (n1 + n2 − 2).

As the t-value increases (standardised difference is further from the center of 0), the less likely it is that the null is true (tail probability decreases), and the smaller the p-value will be.

```python
import numpy as np
from scipy.stats import t

def ttest(group1, group2, two_tailed=True):
    '''
    Run two sample independent t-test
    group 1, 2: numpy arrays for data in each group
    returns: t value, p value
    '''
    
    n1 = len(group1)
    n2 = len(group2)
    var_a = group1.var(ddof=1)
    var_b = group2.var(ddof=1)
    sp = np.sqrt(((n1-1)*var_a + (n2-1)*var_b) / (n1+n2-2))
    t_value = (group1.mean() - group2.mean()) / (sp * np.sqrt(1/n1 + 1/n2))
    df = n1 + n2 - 2
    
    if two_tailed:
        t_prob = t.cdf(abs(t_value), df)
        return t_value, 2*(1-t_prob)
    else:
        t_prob = t.cdf(t_value, df)
        return t_value, 1-t_prob
```

Using scipy function:

```python
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(group1, group2, equal_var=True, alternative='two-sided')
```

**Assumptions**:

- Observations are sampled independently: 
    - Each observation is independent of the others: they don't influence each other, each is counted only once, and each is only given one treatment. Some examples where this might not be true: 1) if is there is some kind of relationship between the customers (e.g. friends and family, who might discuss the treatment and influence each others outcomes); 2) if you can't be certain that each customer is unique (e.g. tracking by http cookie which could be refreshed, bots, or tracking by different IDs per device with multi-device users); and, 3) if supply is shared between the groups (so increased purchases in one group, impacts the ability of the other group to purchase).
- The variance of each group is equal:
    - Identically distributed: The standard t-test uses the variance of one sample to estimate the standard error, so it assumes the variance is equal in both groups. However, if the sample sizes of each group is equal, it's pretty robust to non-equal variances as well. Non-equal variance can occur as a result of the treatment (e.g. a discount may change the composition of bookers, and/or change the spending patterns in a non-equal way for all sub-groups).
    - The sample variance follows a scaled χ2 distribution. For non-normal data, this may not be true, but Slutsky's theorem implies this has little effect on the test statistic with large samples.
- The means follow normal distributions (i.e. the sampling distribution of the estimator is normal). 
    - The central limit theorem (CLT) guarantees conversion to normal for all distributions with finite variance (doesn’t work for power law distributions). However, if the distribution of the observations is non-normal (e.g. skewed, exponential), the speed of convergence might require very large samples for this assumption to be met.
    
In general, the t-test is fairly robust to all but large deviations from these assumptions. However, the equal variance assumption is often violated, so Welch's t-test which doesn't assume equal variance is more commonly used.

### Welch's t-test

Welch's t-test is an adaptation of the t-test which is more reliable when the variance or sample sizes are unequal. Unlike the t-test above, Welch's t-test doesn't use a pooled variance estimate:

$$ t = \frac{\bar{x_1} - \bar{x_2}}{\sqrt{\frac{s_1^2}{n1} + \frac{s_2^2}{n2}}}$$

The degrees of freedom (used to determine the t-distribution and thus critical value):

$$dof = \frac
{(\frac{s_\text{x1}^2}{n1} + \frac{s_\text{x2}^2}{n2})^2}{\frac{s_\text{x1}^4}{n1^2*(n1-1)} + \frac{s_\text{x2}^4}{n2^2*(n2-1)}} $$

Where:
- $$\bar{x_1}$$, $$\bar{x_2}$$ are the sample means,
- n1,n2 are the sample sizes 
- $$s_\text{x1}^2$$, $$s_\text{x2}^2$$ are the sample variances, and $$s_\text{x1}^4$$, $$s_\text{x2}^4$$ are the squared sample variances.

```python
from scipy.stats import t

def welch_t_test(avg_group1, variance_group1, n_group1, 
                 avg_group2, variance_group2, n_group2, two_tailed = True):
    """
    Runs Welch's test to determine if the average value of a metric differs 
    between two independent samples.
    Returns: t value, p value
    """
    
    mean_difference = avg_group2 - avg_group1
    mean_variance_group1 = variance_group1/n_group1
    mean_variance_group2 = variance_group2/n_group2
    
    se =  np.sqrt(mean_variance_group1 + mean_variance_group2)
    t_value = mean_difference / se

    dof_numer = (mean_variance_group1 + mean_variance_group2)**2
    dof_denom = ((mean_variance_group1)**2 / (n_group1-1)) + ((mean_variance_group2)**2 / (n_group2-1))
    dof = dof_numer / dof_denom
    
    if two_tailed:
        t_prob = t.cdf(abs(t_value), dof)
        return t_value, 2*(1-t_prob)

    else:
        t_prob = t.cdf(t_value, dof)
        return t_value, 1-t_prob
```

Using scipy function:

```python
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(group1, group2, equal_var=False, alternative='two-sided')
```

Note, it's not recommended to actually test whether the variance is equal, it's safe to use in place of a t-test even if the variance is equal. However, keep in mind the other assumptions discussed above still apply (normal sampling distribution, independence).

### z-test

For completeness, let's briefly cover the z-test. The z-test compares the standardised difference between the means of two independent samples to the standard normal distribution. Like before, based on how often a z-statistic that large or larger is observed when the the null hypothesis is true (no difference), we decide to accept or reject the null hypothesis.

$$ Z = \frac{\bar{x_1} - \bar{x_2}}{sqrt(\sigma_\text{x1}^2/n1 + \sigma_\text{x2}^2/n2)}$$

Where the denominator is the standard error for the mean difference. Note, we need to know the population variances for the z-test, so it's not often used in practice.

## Common issues:

We've covered three common tests for comparing the means of two independent samples: t-test, Welch's t-test, and the z-test. In practice, Welch's t-test is most commonly used when comparing the means of two independent samples. However, keep in mind they all share some common issues:

- Given they assume the means follow normal distributions, larger samples are required for skewed and non-normal distributions (based on CLT, they will eventually converge). Before applying these tests, the metrics should be validated. You may also want to consider methods to improve the power e.g. variance reduction via regression adjustment.
- Outliers: extreme values far from the mean will explode the variance in the denominator, causing the t statistic to become very small and decreasing the power. One option is to winsorize the values first (capping at some percentile). 

References: [Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test)
