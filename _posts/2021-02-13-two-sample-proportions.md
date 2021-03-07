---
layout: post
title: "Hypothesis testing: Two sample tests for proportions"
mathjax: true
---

This post covers the most commly used statistical tests for comparing a binary (success/failure) metric in two independent samples. For example, imagine we run an A/B experiment where our primary goal is to increase the conversion rate. Visitors can either make a purchase (convert), or not. Therefore, our metric is 1 if they convert, else 0. To test whether there was any significant effect on conversion, we compare the proportion of visitors who converted in each sample. 

Other examples of proportion metrics: proportion of visitors who click on something, proportion who cancel, proportion who contacted customer service. 

## Hypothesis

Much like with continuous metrics, our two-sided hypothesis looks something like:

- Ho: p1-p2 = 0
- Ha: p1-p2 ≠ 0 (not equal)

With the following alternates for a one-sided test:

- Ha: p1 - p2 > 0 (p1 is greater than p2)
- Ha: p1 - p2 < 0 (p1 is less than p2)

## General notation:

- $$\hat{p_1}$$, $$\hat{p_2}$$ are the sample proportions (p1 and p2 are the unknown population proportions).
- n1,n2 are the sample sizes 

## Statistical tests:

### z-test (normal approximation)

If the sampling distribution for the difference in proportions is normal, we can use a z-test:

- The sampling distribution of $$\hat{p_1}$$ is binomial, with p1 probability of success and n1 trials. The mean is p1, the variance is p1(1 -p1)/n1, and the standard deviation is the square root of the variance. Same goes for p2.
- These binomial distributions can be approximated by a normal distribution. The difference, p̂1 - p̂2, is a linear combination of the two random variables, and so also has a normally distributed sampling distribution. 
- The sampling distribution of p̂1 - p̂2 has mean p1-p2, and variance (additive): p1(1 - p1)/n1 + p2(1 - p2)/n2. Using the sample proportions, the variance is estimated as: p̂1(1 - p̂1)/n1 + p̂2(1 - p̂2)/n2.

$$z = \frac{(\hat{p_1} - \hat{p_2})}{\sqrt{p(1-p)(\frac{1}{n1} + \frac{1}{n2})}}$$

$$p = \frac{x_1+x_2}{n_1+n_2}$$ 

Where x1 and x2 are the number of successes in each group. Like the pooled estimate used in the two-sample t-test, the result for p will be between p̂1 and p̂2.

```python
import numpy as np
import scipy.stats.distributions as dist

def z_test_2_sample_proportions(x1, x2, n1, n2, two_tailed=True):
    '''
    Calculate the test statistic for a z-test on 2 proportions from independent samples
    x1, x2: number of successes in group 1 and 2
    n1, n2: total number of observations in group 1 and 2
    Returns: test statistic (z), and p-value 
    '''
    avg_p = (x1 + x2) / (n1 + n2)
    z_val = (x1/n1 - x2/n2) / np.sqrt(avg_p * (1-avg_p) * (1/n1 + 1/n2))
    z_prob = dist.norm.cdf(-np.abs(test_stat))

    if two_tailed:
        return z_val, 2*z_prob

    else:
        return z_val, z_prob
```

Using [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html):

```python
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

success_cnts = np.array([x1, x2])
total_cnts = np.array([n1, n2])
test_stat, pval = proportions_ztest(count=success_cnts, nobs=total_cnts, alternative='two-sided')
print('Two sided z-test: z = {:.4f}, p value = {:.4f}'.format(test_stat, pval))
```

The wald statistic is an alternative test statistic, which is useful for calculating the confidence interval:

$$\text{Wald test statistic} = \frac{(\hat{p_1} - \hat{p_2})}{\sqrt{\frac{\hat{p_1}(1-\hat{p_1})}{n_1} + \frac{\hat{p_2}(1-\hat{p_2})}{n_2}}}$$

$$\text{approximate CI} = \hat{p_1} - \hat{p_2} \pm Z_\text{critical} * \text{Wald test statistic}$$

**Assumptions**

- Samples are independent and randomly selected.
- The binomial distribution is closely approximated by the gaussian distribution. This is true for large samples.
- Rule of thumb: expected successess and failures in both samples at least 10. If there are fewer, an exact test can be used.

<br>

### Chi-square test of homogeneity

An alternative to the z-test is the chi-square test. However, it's been shown that the 2x2 chi-square homogeneity test is equivelant to the z-test of two proportions (see [here](http://rinterested.github.io/statistics/chi_square_same_as_z_test.html)). 

$$\chi^2 = \sum{\frac{(O_i - E_i)^2}{E_i}}$$

**Expected count:** 

The expected count for a cell is the joint probability of the groups * total count. E.g. If the joint probability is 0.3 and we have a sample of 100, then we expect 0.3*100=30 for that cell. Worked example:

- The probability of annual subscriber + AUS (assuming independence) = P(status=AS, country=AUS) = P(AS)*P(AUS) = 25,006/250,000 * 24,963/250,000 = 624224778/62500000000 = 0.00999
- We multiply this by the total count to get the expected count for status=AS & country=AUS = 250K * 624,224,778 / 62,500,000,000 = 2496.9
- Pulling that together we see the expected count = (250,000 * 25,006 * 24,963) / (1 * 250,000 * 250,000), and we can cancel out at 250,000 in the numerator and denominator, leaving: 25,006*24,963 / 250,000.
- In other words, the expected count simplifies to = count(AS) * count(AUS) / total count

![Contingency table](/assets/contingency_table.png)

**Chi-square statistic:** 

For each cell (combination of groups), calculate the squared difference between the observed and expected counts and divide by the expected to standardise = (observed-expected)^2/expected. Sum these values to get the Chi-square statistic.

**Python implementation:**

```python
from statsmodels.stats.proportion import proportions_chisquare

success_cnts = np.array([x1, x2])
total_cnts = np.array([n1, n2])
chi2, p_val, cont_table = proportions_chisquare(count=success_cnts, nobs=total_cnts)

print('Chi-square statistic = {:.2f}'.format(chi2))
print('p-value = {:.4f}'.format(p_val))
```

### Alternatives

Finally, the [g-test](https://en.wikipedia.org/wiki/G-test) and [Fisher's exact](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) test are more precise options. However, these converge with the chi-square test with large samples.
