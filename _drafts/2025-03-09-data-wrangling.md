---
layout: post
title: "Data wrangling for analysis"
mathjax: true
---

Cover the standard cleaning need to do:

- missing data
- outliers
- reshaping and merging data
- converting categorical features (OHE, target encoding, ...) etc.

Checking data: 
- Summary tables etc.



Dummy variables are categorical variables we’ve encoded as binary columns. For example, suppose you have a gender variable that you wish to include in your model. This variable is encoded into 3 categories: male, female and other genders. Since our model only accepts numerical values, we need to convert this category to a number. In linear regression, we use dummies for that. We encode each variable as a 0/1 column, denoting the presence of a category. We also leave one of the categories out as the base category. This is necessary since the last category is a linear combination of the others. Put it differently, we can know the last category if someone gives us information on the others. In our example, if someone is neither female nor other genders, we can infer that the person’s category is male.


Example post:

https://exploration.stat.illinois.edu/learn/Understanding-and-Wrangling-Data/Measurement-Errors/ 
