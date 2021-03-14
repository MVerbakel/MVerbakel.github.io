---
layout: post
title: "Confidence intervals"
mathjax: true
---

What is a confidence interval

Confidence interval for differences vs estimates

One vs two sided

Ratio metrics

Bootstrapped CI


Standard Error of Mean: This is the estimated standard deviation for the distribution of sample means for an infinite population. It is the sample standard deviation divided by the square root of sample size.


## Sampling distribution

So what if I repeat that exercise many times? Intuitively, the estimates should vary around the true estimate, so the mean of that distribution will converge to the true mean value as the number of estimates increases (this is the law of large numbers). The distribution of estimates (means in this case) is what we call the sampling distribution of the estimator. Below I show what this looks like if we plot the mean estimates from 30 days (each time weighing 10 kangaroos), 100 days, and 1000 days. As the number of estimates increases, it looks more and more normal (bell curve). This is what the Central Limit Theorem (CLT) tells us: as n increases, the distribution converges to the normal distribution (with the same mean as the population, but a smaller variance). 

![Sampling distributions](/assets/sampling_distribution_example.png)

This means that some days I'll get a very low estimate, others a high estimate, but on average it's around 46kg. The fact that it converges to a normal distribution centered on the mean is an extremely useful property. I'll explain why next. First, note the CLT does come with some assumptions:
- the data is a random sample;
- each value is independent (they don't influence each other);
- the sample size is not >10% if the sampling is without replacement; and,
- the sample size is sufficiently large.

## Standardising

Standardising the sampling distribution allows us to talk more generally about the probability of different estimates. To create the standardised values, Z, we take the difference between each x value and the mean, and divide by the standard deviation:

$$ Z = \frac{x - \mu}{\sigma} $$

Z has a mean of 0 and standard deviation of 1, and is normally distributed (as X was normally distributed). Plotting the values of Z, we can easily see the 68-95-98 rule: 68% is within 1 standard deviation of the mean, 95% is within 2 standard deviation of the mean, and 99.7% is within 3 standard deviation of the mean.

![Sampling distributions](/assets/standard_normal_distribution.png)

## Summarising the sampling distribution

There are two common ways to summarise the sampling distribution:
- **1) The standard error:** a measure of the variability in the sampling distribution, or how far off we expect an estimate to be off on average. As n increases, the standard error decreases. Not to be confused with standard deviation (variability in a measured quantity). Note, the population standard deviation ($\sigma$) is not usually known, so it's usually estimated with the sample standard deviation (s).

$$ SE_\text{mean} = \frac{\sigma}{\sqrt{n}}$$

- **2) Confidence interval (CI):** a range that includes a given fraction of the sampling distribution. E.g. the 95% CI is the range from the 2.5th to 97.5th percentiles (5% split on either side). When explaining to a non-technical audience, we'll normally say something like we're reasonably confident that the true value is somewhere within the CI range, or this is a range of plausible values for the true value. However, don't misinterpret the confidence. We're measuring how reliable our estimate is, based on how much estimates would vary across repeated experiments. So there isn't a 95% probability the actual value falls within the confidence interval. Rather, 95% of the time (if we repeat our sampling process), we get a confidence interval that contains the true value. In the other 5% the true value is not in the CI at all (i.e. 1/20 experiments I'm wrong and return a CI that doesn't contain the true value). 

$$ CI_\text{mean} = \bar{x} \pm \text{Margin of error} $$

$$ \text{Margin of error} = \text{critical value} * SE = z \frac{s}{\sqrt{n}} $$

Where 's' is the sample standard deviation, and the critical value 'z' is based on our selected confidence. E.g. z=1.96 for 95% confidence. The z value comes from looking at the standardised sampling distribution and calculating how many standard deviations do we need to have the area under the curve cover 95% of values. E.g. we saw earlier that for 65% we would only need +/-1 standard deviation, and for 95% we need ~2 (it's actually 1.96 to be precise, which is the critical z above). 

## Confidence interval for the mean estimate

Now, coming back our example. In reality, we don't want to repeat the sampling many times. We want to take that first estimate from one day of weighing 10 kangaroos (45.3kg), and understand how confident we should be that this is accurate. To do that we'll create a confidence interval based on what we learned above:

```python
import numpy as np
np.random.seed(26)

# Create a sample of 10 weights based on a normal distribution (mu=46, sigma=3)
single_day_measurements = np.random.normal(46, 3, 10)
sample_mean = np.mean(single_day_measurements)
sample_std_dev = np.std(single_day_measurements, ddof=1)
n = len(single_day_measurements)

se = sample_std_dev / np.sqrt(n)
moe = 1.96 * se
ci_lower, ci_upper = (sample_mean - moe), (sample_mean + moe)

print('Estimated mean = {:.2f}'.format(sample_mean))
print('Estimated standard deviation = {:.2f}'.format(sample_std_dev))
print('Standard error = {:.3f}'.format(se))
print('Margin of error = {:.3f}'.format(moe))
print('95% CI = ({:.2f}, {:.2f})'.format(ci_lower, ci_upper))

Estimated mean = 45.31
Estimated standard deviation = 2.91
Standard error = 0.921
Margin of error = 1.804
95% CI = (43.51, 47.12)
```

So based on this sample, we estimate the population mean is somewhere in the range 43.5 to 47.1kg with 95% confidence. To make sure the interpretation is clear, the plot below repeats this exercise 20 times and shows the resulting confidence intervals compared to the true population mean (not known in reality). Notice that 19 of the confidence intervals overlap with the mean, but 1 does not. This matches our expectation that 19/20 times (95%) our estimation process works and produces a confidence interval that contains the mean, but 1/20 times (5%) it does not. 

![Confidence intervals](/assets/confidence_interval_interpretation.png)

## Complete Python code and formulas: estimating a population mean

Note, if the population standard deviation isn't known and the sample size is small (<30), we can use the t-distribution to determine the critical value instead of z. Like the normal distribution, the t-distribution is symmetric and bell shaped, but it has heavier tails (more values at the extremes, so values far from the mean more common than for z). However, for large n the t-distribution becomes very close to the z distribution, so it's a safe bet in general to pick t instead of z. Below I'll show how to use both methods in Python.
