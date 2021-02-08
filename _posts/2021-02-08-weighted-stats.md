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

Reference: [Wikipedia](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
