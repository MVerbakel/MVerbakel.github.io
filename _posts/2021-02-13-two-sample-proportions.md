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

Another option is the chi-square test of homogeneity. The chi-square test takes the observed frequencies in a contingency table (crosstab of the outcome levels vs sample groups), and compares them to what we would expect to see if there was no difference (null hypothesis). In the case of an experiment, if the treatment had no effect, we expect to see the proportion in each outcome level approximately the same for both samples. Note, this is just one of the three types of comparison for the chi-square test: 1. goodness of fit for comparing an observed and theoretical distribution; 2. test of homogeneity (this one) to compare two or more groups on the same categorical variable; and, 3. test of independence two compare two variables. However, they all use the same math for the test-statistic.

$$\chi^2 = \sum{\frac{(O_i - E_i)^2}{E_i}}$$

Note, it's been shown that the 2x2 chi-square homogeneity test is equivelant to the z-test of two proportions (see [here](http://rinterested.github.io/statistics/chi_square_same_as_z_test.html)). However, the chi-square test can also handle cases with more than 2 levels in the outcome, or more than two populations.

**Expected counts:** 

The expected count for a cell is the joint probability of the groups * total count. E.g. If the joint probability is 0.3 and we have a sample of 100, then we expect 0.3*100=30 for that cell. Worked example:

|  Group  | Yes |  No |  Total  |
| ------- | --- |  -- |  -----  |
| control | C_Y | C_N | C |
| variant | V_Y | V_N | V |
| totals  | Y | N | T |

Based on this we can calculate the expected counts:
- Expected C_Y = E_C_Y = P(control) * P(Yes) * Total = C/T * Y/T * T = C*Y*T / 1*T*T = C*Y / T
- Expected C_N = E_C_N = C*N / T
- Expected V_Y = E_V_Y = V*Y / T
- Expected V_N = E_V_N = V*N / T

**Chi-square statistic:** 

For each cell (combination of groups), calculate the squared difference between the observed and expected counts and divide by the expected to standardise = (observed-expected)^2/expected. Sum these values to get the Chi-square statistic. The larger the difference between the observed and expected, the larger the test-statistic. 

The test statistic approximately follows the chi-square distribution under the null hypothesis. Therefore, we can compare our test value to the chi-square distribution (adjusting the shape based on the degrees of freedom calculation below), and estimate the probability of observing such a value (or larger) when the null is true (p-value). 

$$\text{degrees of freedom} = (\text{nr rows} - 1) * (\text{nr columns} - 1)$$

For a 2x2 table, the degrees of freedom = (2-1) * (2-1) = 1.

**Python implementation:**

```python
from statsmodels.stats.proportion import proportions_chisquare

success_cnts = np.array([x1, x2])
total_cnts = np.array([n1, n2])
chi2, p_val, cont_table = proportions_chisquare(count=success_cnts, nobs=total_cnts)

print('Chi-square statistic = {:.2f}'.format(chi2))
print('p-value = {:.4f}'.format(p_val))
```

**Assumptions**

- Samples are independent and randomly selected. Each observation fits in one and only one cell.
- The expected counts should be 5 or more in at least 80% of the cells, and no cell has an expected count <1. If they are not, [Fisher's exact](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) test is recommended (a more accurate version of this test). 

### G-test

Finally, the [g-test](https://en.wikipedia.org/wiki/G-test) is another alternative. The g-test is very similar to the chi-squares test, as the chi-squared test is actually an approximation of the g-test. As with the chi-square test, we take a contingency table, and calculate by how much the observed and expected values differ. The null hypothesis being that the proportions of the outcome levels are the same for both samples.

**Expected counts:**

The expected counts are calculated the same way as the chi-squared test above, but the test statistic is different.

**G test statistic** (based on presentation by [Tilly](https://elem.com/~btilly/effective-ab-testing/)):

For each cell we calculate the g-value and the g-statistic is the sum of these values times 2:

$$g = 2 * \sum{O_i * \log{\frac{O_i}{E_i}}}$$

**Python implementation:**

```python
from scipy.stats import chi2
import numpy as np

def g_test(n1, n2, successes_1, successes_2):
    """
    Performs a g-test comparing the proportion of successes in two groups.
    Returns: test statistic, p value
    """

    failures_1 = n1 - successes_1
    failures_2 = n2 - successes_2
    total = n1 + n2

    expected_control_success = n1 * (successes_1 + successes_2) / total
    expected_control_failure = n1 * (failures_1 + failures_2) / total
    expected_variant_success = n2 * (successes_1 + successes_2) / total
    expected_variant_failure = n2 * (failures_1 + failures_2) / total

    g1 = successes_1 * np.log(successes_1/expected_control_success)
    g2 = failures_1 * np.log(failures_1/expected_control_failure)
    g3 = successes_2 * np.log(successes_2/expected_variant_success)
    g4 = failures_2 * np.log(failures_2/expected_variant_failure)

    g_test_stat = 2 * (g1+g2+g3+g4)
    pvalue = 1 - chi2.cdf(g_test_stat, 1)

    return g_test_stat, pvalue
```

Alternatively, there's a scipy function:
```python
from scipy.stats import chi2_contingency

g, p, dof, expctd = chi2_contingency(contingency_table, lambda_="log-likelihood", correction=False)

print("g test statistic={:.4f}, dof={}, p-value={:.4f}".format(g, dof, p))
```

**Assumptions:**

- Samples are independent and randomly selected. Each observation fits in one and only one cell.
- All observed counts (cells) are at least 10. 

**Yates Continuity Correction**

Both the chi-square test and g-test err on the small side, so Yates suggested adjusting all observed values 0.5 towards the expected values. This improves accuracy slightly. Set correction=True in the scipy function above for this correction.

### Final thoughts:

- The z-test for two proportions is commonly used in business as samples are large. The-chi-squared test is also pretty common, but as noted, equivelant to the z-test in the 2x2 case.
- Some argue the g-test is better as it's faster and has less assumptions than the z-test.
- However, with large samples (>~10K), the results of the z-test, chi-squared test, and g-test will all converge. So it mostly comes down to the field/company in practice (choose whatever is familiar to most).  
