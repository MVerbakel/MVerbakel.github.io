---
layout: post
title: "Weighted statistics"
mathjax: true
---

Sometimes the sample data we have doesn't represent the population well. For example, maybe you run a survey and the response rate is higher for males than females. If the variables we want to analyse differ by the unbalanced groups (e.g. gender), then all our summary statistics will be biased. To correct this we can use weighting.

### Create sample

In our example data we have 6 measurements from males and 4 from females (so 60% male). Let's say in the population we know it's actually 50% male, so they're over-represented in our sample. If the measurement differs by gender, the mean will be biased towards males. However, we can correct this be weighting the measurements by the actual/target proportion as below.

```python
test_sample = pd.DataFrame({'gender': ['male']*6 + ['female']*4,
                            'gender_bin': [1]*6 + [0]*4,
                            'measurement': [2.5]*4 + [1]*2 + [4.5]*4
                           })

current_male_prop = 0.6
population_male_prop = 0.5
male_weight = population_male_prop/current_male_prop

current_female_prop = 0.4
population_female_prop = 0.5
female_weight = population_female_prop/current_female_prop

test_sample['weights'] = test_sample['gender'].map(lambda x: female_weight if x=='female'
                                                   else male_weight)

print('Unweighted mean = {:.3f}'.format(test_sample['measurement'].mean()))
print('Male weight = {:.2f}, female weight = {}'.format(male_weight, female_weight))
Unweighted mean = 3.0
```
```python
Male weight = 0.83, female weight = 1.25
```

### Weighted mean

The weighted mean is just the values multiplied by the weights, and divided by the sum of the weights:

$$ \bar{x}_\text{weighted} = \frac{\sum\limits_{i=1}^n (x_iw_i)}{\sum\limits_{i=1}^n(w_i)}$$

```python
test_sample['weights'] = test_sample['gender'].map(lambda x: female_weight if x=='female'
                                                   else male_weight)

weighted_measure_sum = sum(test_sample['measurement'] * test_sample['weights'])
sum_weights = sum(test_sample['weights'])

weighted_mean = weighted_measure_sum / sum_weights
print('Weighted mean = {:.3f}'.format(weighted_mean))
```
```python
Weighted mean = 3.250
```

### Weighted variance & standard deviation

For the variance we want to compare the actual values to the corrected mean (weighted mean), and then we can weight the squared differences and divide by the sum of weights to get their average with the correct proportions. 

$$ \hat\sigma^2_w = \frac{\sum\limits_{i=1}^n w_i(x_i - \mu^*)^2}{\sum\limits_{i=1}^n(w_i)}$$

```python
squared_diffs = np.array([(x - weighted_mean)**2 for x in test_sample['measurement']])
sum_squared_diffs = np.dot(squared_diffs.T, test_sample['weights'].values)
weighted_variance = sum_squared_diffs / sum_weights
weighted_std_dev = weighted_variance**0.5
print('Weighted variance = {:.3f}'.format(weighted_variance))
print('Weighted standard deviation = {:.3f}'.format(weighted_std_dev))
```
```python
Weighted variance = 1.812
Weighted standard deviation = 1.346
```

### Weighted t-test

We can now use our weighted mean and weighted standard deviation directly in a t-test, just remember n or nobs is sum(weights). Let's assume we have two samples (control and variant), and want to compare their weighted means (assuming both were equally impacted by the imbalance).

#### Create a second sample

For convenience, I'll just use the same data, but add some difference to the measurements.

```python
import pandas as pd
import numpy as np

target_proportion = 0.5

test_sample_b = pd.DataFrame({'gender': ['male']*6 + ['female']*4,
                              'gender_bin': [1]*6 + [0]*4,
                              'measurement': [3.5]*4 + [1]*2 + [4.6]*4
                              })

male_prop_b = target_proportion / test_sample_b['gender_bin'].mean()
female_prop_b = target_proportion / (1 - test_sample_b['gender_bin'].mean())

test_sample_b['weights'] = test_sample_b['gender'].map(lambda x: female_prop_b if x=='female'
                                                       else male_prop_b)
```

#### Calculate the weighted mean and standard deviation

```python
def weighted_statistics(values, weights):
    """
    values: numpy array of measurements
    weights: numpy array of weights to apply to values (should be same length)
    Returns: weighted mean, weighted standard deviation, and the sum of the weights
    """
    
    sum_of_weights = sum(weights)
    weighted_sum_of_values = np.dot(values.T, weights)
    weighted_mean = weighted_sum_of_values / sum_of_weights
    
    diffs_from_mean = np.array([x-weighted_mean for x in values])
    sum_squares = np.dot((diffs_from_mean**2).T, weights)
    weighted_stddev = np.sqrt(sum_squares / sum_of_weights)
    
    return weighted_mean, weighted_stddev, sum_of_weights

weighted_mean_a, weighted_stddev_a, sum_of_weights_a = weighted_statistics(values=np.array(test_sample_a['measurement']), 
                                                                           weights=np.array(test_sample_a['weights'])
                                                                          )

weighted_mean_b, weighted_stddev_b, sum_of_weights_b = weighted_statistics(values=np.array(test_sample_b['measurement']), 
                                                                           weights=np.array(test_sample_b['weights'])
                                                                          )

print('A: weighted mean = {:.3f}, weighted std dev = {:.3f}'.format(weighted_mean_a, weighted_stddev_a))
print('B: weighted mean = {:.3f}, weighted std dev = {:.3f}'.format(weighted_mean_b, weighted_stddev_b))
```
```python
A: weighted mean = 3.250, weighted std dev = 1.346
B: weighted mean = 3.633, weighted std dev = 1.276
```

#### Apply Welch's t-test

Ho: mean of A = mean of B
Ha: mean of A != mean of B

```python
from scipy.stats import t

mean_variances = [weighted_stddev_a**2/sum_of_weights_a, weighted_stddev_b**2/sum_of_weights_b]
t_value = (weighted_mean_b - weighted_mean_a)/np.sqrt(sum(mean_variances))
t_df = int( sum(mean_variances)**2 / 
            ( ( mean_variances[0]**2 / (sum_of_weights_a - 1) ) 
            + ( mean_variances[1]**2 / (sum_of_weights_b - 1) ) ) 
            )  

t_prob = t.cdf(abs(t_value), t_df)
p_val = 2*(1-t_prob)
print('t-statistic = {:.3f}, p-value = {:.3f}'.format(t_value, p_val))
```
```python
t-statistic = 0.653, p-value = 0.522
```

#### Alternative

Alternatively, there are a number of statistical packages that can handle weights. E.g. the above is almost identical to the approach used in the package statsmodels.stats.weightstats:

'''python
from statsmodels.stats.weightstats import ttest_ind

tstat, pval, dof = ttest_ind(test_sample_b['measurement'],
                             test_sample_a['measurement'],
                             usevar='unequal',
                             weights=(test_sample_b['weights'], test_sample_a['weights']),
                             alternative='two-sided'
                            )

print('t-statistic = {:.3f}, p-value = {:.3f}'.format(tstat, pval))
'''
```python
t-statistic = 0.620, p-value = 0.543
```


References: [Wikipedia](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance), [statsmodels](https://www.statsmodels.org/stable/_modules/statsmodels/stats/weightstats.html)

