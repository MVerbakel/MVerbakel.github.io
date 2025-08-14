---
layout: post
title: "Probability refresher"
mathjax: true
---

The goal of probability theory is to quantify uncertainty. Once an event occurs, the outcome is known (e.g. it rained today or it didn’t). However, before an event happens, probability allows us to assess the likelihood of various potential outcomes (e.g. the chance that it will rain). Depending on the relationship between events, probability rules help us calculate the likelihood of a given outcome in a straightforward manner. A fundamental concept in probability is that of random variables, which assign numerical values to the outcomes of random processes. These random variables can take different forms: discrete, where the outcomes are countable (such as the number of heads in a series of coin flips), or continuous, where the outcomes can take any value within a given range (such as the height of a person). Understanding and applying probability theory equips us to make informed decisions based on the inherent uncertainty in real-world phenomena.

## Random Variables

A random variable (also called a stochastic variable) is a function that assigns numerical values to the outcomes of a random event. These outcomes are drawn from a sample space, which is the set of all possible results. For instance, when flipping a fair coin, there are two equally likely outcomes: heads or tails. If we define a random variable X that assigns 1 to heads and 0 to tails, the probabilities for each are: P(H)=P(T)=½, which add to 1. Random variables can be:

- Discrete: Taking values from a finite or countable set, such as the number of heads in a series of coin flips. The probability distribution for discrete random variables is described by a probability mass function (PMF).
- Continuous: Taking values from an uncountable range, such as the height of a person, which can vary continuously. For continuous variables, the probability distribution is described by a probability density function (PDF).
- Mixture of both: Sometimes random variables have a combination of discrete and continuous components.

In statistics, random variables are usually considered to take real number values, which allows us to define important quantities like the expected value (mean), and variance. 

## Distributions

A probability distribution lists all possible outcomes of a random variable along with their corresponding probabilities (must sum to 1). Rather than listing all possible probabilities, we can define a function f(x) that gives the likelihood of certain outcomes in a random process. Some of the most common probability distributions are shown below:

<img src="/assets/probability_distributions.png" alt="distributions" width="100%"/>

Another one that comes up is the **power law** distribution: one quantity varies as a power of another ($$y=x^\text{\alpha}$$, where y is the dependent, x is the independent and $$\alpha$$ is a constant). It has heavy tails (few large values dominate, while many smaller values are common) and scale invariance (the system behaves similarly at different scales, so the pattern remains consistent even when zooming in or out). The extreme values or outliers (like the largest city or the richest person) have a disproportionate effect compared to the "average" values, making it important to understand and account for these distributions in analysis and modelling. Some examples: 
- **Zipf's law** is a specific type of power law that describes how wealth, population, or even word frequency is distributed. In Zipf’s law, the rank r of an item is inversely proportional to its frequency f, meaning the most frequent item is far more common than the next most frequent, and so on. E.g. the largest city might have 10 million people, the second-largest 5 million, the third-largest 2 million, and so on. The number of cities with these population sizes follows a power law distribution.
- **Scale-free networks** are characterized by a small number of highly connected nodes (hubs) and many nodes with fewer connections. These networks follow a power law distribution in terms of their degree (the number of connections each node has). Most nodes have few connections, but a few nodes have disproportionately large numbers of connections. Examples include social networks (where a few individuals have many followers) or the internet (where a few websites receive the majority of traffic).

When plotted, it typically produces a curve that starts high and quickly tapers off, which can make it difficult to directly see the relationship. However, if we plot the logarithm of both the variable and the frequency, the relationship becomes linear:

<img src="/assets/power_law_examples.png" alt="power law" width="100%"/>

## PMF / PDF / CDF

The distributions shown above are either a PMF or PDF depending on the outcome type:

<img src="/assets/pmf_pdf_cdf.png" alt="PMF vs PDF" width="100%"/>

| **Metric**                                | **Formula**                                                                                                         | **Description**                                                                                                               |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Probability Mass Function (PMF)**       | $$P(X = x)$$                                                                                                       | Describes the probability distribution of a discrete random variable. The PMF gives the probability that a discrete random variable is exactly equal to some value. For example, in a coin toss, the probability of heads or tails. |
| **Probability Density Function (PDF)**    | $$f(x) = \frac{d}{dx} P(X \leq x)$$                                                                                 | Describes the probability distribution of a continuous random variable. The PDF defines the probability of the random variable falling within a particular range of values, and its integral over an interval gives the probability. The total area under the curve equals 1. |
| **Cumulative Distribution Function (CDF)**| $$F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) dt$$                                                                  | Gives the probability that a random variable is less than or equal to a certain value. For a discrete variable, it’s the sum of the probabilities of all outcomes less than or equal to \(x\). For a continuous variable, it’s the area under the PDF curve up to \(x\). |

## Moments

Moments are used to describe the shape of the distribution. Each "moment" provides different information about the distribution’s characteristics. Here’s the first few moments:

- Zeroth moment: The total probability, which is always 1.
- First moment: The expected value (mean), which gives the central location of the distribution.
- Second central moment: The variance, which measures the spread of the data around the mean.
- Third standardized moment: The skewness, which describes the asymmetry of the distribution.
- Fourth standardized moment: The kurtosis, which measures the "tailedness" or how extreme the values in the distribution are.

## Expected Value

The average value over the long run. We can find an expected value by multiplying each numerical outcome by the probability of that outcome, and then summing those products together. Consider a game where we have a $$4/10$$ chance of winning $2, a $$4/10$$ chance of losing $5, and a $$2/10$$ chance of winning $10. The EV is:

$$4/10 * 2 + 4/10 * -5 + 2/10 * 10 = 0.8-2+2=0.8$$. 

So on average, we expect to win 80 cents per game in the long run. In statistics, a similar concept is the population mean, which is often estimated from the sampling distribution—the probability distribution of outcomes obtained from many samples.

## Set notation

A set is simply a collection of distinct elements. In probability, we often deal with sets to define events and sample spaces. The sample space (denoted as S) is the set of all possible outcomes (e.g. S={H, T}). An event is any subset of the sample space (e.g. the event that the coin lands on heads can be represented as A={H}). When describing relationships between events, we use set notation:

<img src="/assets/set_notation.png" alt="sets" width="100%"/>

## Marginal (unconditional) probability

This is just the probability of A (P(A)).

## Complement rule

The probability that something doesn't happen is 1 minus the probability that it does happen: $$P(A^c)= 1-P(A)$$. The probability of something not happening is often easier to compute, so we can take advantage of that.

Example: Assuming the days are independent, what is the probability of at least 1 day of rain in the next 5 days if the P(rain)=0.2 each day? 
- The complement of "at least one day of rain" is "no rain at all" over the 5 days.
- Since the days are independent, the probability of no rain on a given day is 1−P(rain)=1−0.2=0.8. For 5 independent days with no rain, the probability of no rain on all 5 days is simply the product of the probabilities for each day: $$P(no rain)= 0.8^5$$.
- Therefore, the probability of at least one day of rain is $$P(rain)= 1 - 0.8^5$$.
- This is much simpler than directly computing it: summing the probability of rain on exactly 1 day, exactly 2 days, and so on, up to 5 days, then accounting for overlaps and intersections (the joint probabilities).

## Conditional Probability

The probability of event A occurring, given that event B has already occurred. The formula for conditional probability is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Divide joint probability of A and B, by the marginal probability of B.

## Impact of dependence

**Independent events** are events where the occurrence of one event does not affect the occurrence of another. For example, in two independent coin flips, the outcome of the first flip does not affect the outcome of the second flip.
- Addition rule (A or B): If mutually exclusive (can't happen at the same time) then $$P(A \cup B) = P(A) + P(B)$$, if they're not mutually exclusive (e.g. getting H on a coin and rolling a 4): $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$. The P(A∩B) is subtracted once to avoid double counting it in A and B.
- Multiplication rule (A and B): $$P(A \cap B) = P(A) \times P(B)$$

**Dependent events** are events where the occurrence of one event affects the probability of the other event. An example of this is drawing cards without replacement from a deck. If you draw a card and don't replace it, the probability of drawing another specific card changes.
- Multiplication rule (A and B): $$P(A \cap B) = P(A) \times P(B \mid A)$$

## Odds

**Odds** are a different way of expressing the likelihood of an event. While probability is the ratio of favorable outcomes to total outcomes, odds are typically expressed as the ratio of favorable outcomes to unfavorable outcomes. For example, if you have a 1 in 5 chance of winning, the odds are 1:5. Odds magnify probabilities, so they allow us to compare probabilities more easily by focusing on large whole numbers. E.g. comparing 0.01 to 0.005 it's hard to see one is twice as large, but comparing 99 to 1 versus 199 to 1 makes it easier to see the relative difference.

## Combinations and permutations

Combinations and permutations are used to count the number of ways outcomes can be arranged:
- **Combinations** count the number of ways to select a subset of objects where the order does not matter. We write nCk (n choose k) to represent the number of ways to form a combination of k items from n items. 100C100=1 as it's the same as selecting all.

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

- **Permutations** count the number of ways to arrange a set of objects where the order matters. We can write nPk, which the ways we can form k items from n items, counting different orderings of the same set. This will be greater than or equal to nCk.

$$P(n, k) = \frac{n!}{(n-k)!}$$

## Bayes' Theorem

Bayes' Theorem is the consequence of rearranging the conditional probability formula:

$$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$$

Where:
- $$P(A \mid B)$$ is the posterior probability (what we want to find),
- $$P(B \mid A)$$ is the likelihood (how likely we are to observe the evidence if the hypothesis is true),
- P(A) is the prior probability (our initial belief),
- P(B) is the marginal likelihood (the total probability of the evidence).

It can also be written as: posterior= likelihood*prior/evidence.

**Example**: A box contains 8 fair coins (heads and tails) and 1 coin which has heads on both sides. I select a coin randomly and flip it 4 times, getting all heads. If I flip this coin again, what is the probability it will be heads?

- Fair coin: $$P(H \mid fair)=0.5$$
- Biased coin: $$P(H \mid not fair)=1$$
- Prior probability of selecting a fair coin: P(fair) = 8/9
- Conditional probability of 4 H given it's a fair coin: $$P(4 heads \mid fair) = 0.5^4 = 0.0625$$
- Conditional probability of 4 H given it's an unfair coin: $$P(4 heads \mid not fair) = 1$$
- Total probability of 4 H: P(4 heads) = 8/9*0.0625 + 1/9*1
- Now, use Bayes' Theorem to calculate the probability that the coin is fair, given that we observed 4 heads: $$P(fair \mid 4 heads) = P(4 heads \mid fair)*P(fair) / P(4 heads) = 0.0625 * 8/9 / 8/9*0.0625 + 1/9*1 = 1/3$$
- Using this we can calculate the probability that the next flip will be heads: $$P(H) again = 1/3 * P(H \mid F) + 2/3 * P(H \mid not fair) = 1/3 * 1/2 + 2/3 * 1 = 5/6$$

## Simpson's paradox

Simpson's paradox is a counterintuitive phenomenon that occurs when trends observed in different groups of data reverse when the data is combined. Meaning the aggregated data may lead to a different conclusion. This is a specific type of confounding, where the confounding variable might be the group size or way the data is aggregated across different groups. 

**Example**: Consider two doctors performing two types of surgeries. Doctor A has a higher success rate than Doctor B for both types of surgery individually, but performs the more difficult surgery more often (which has a lower succes rate for both doctors), resulting in a lower overall success rate:
- Doctor A has a 95% success rate for surgery type 1 (easier surgery) and a 50% success rate for surgery type 2 (more difficult surgery).
- Doctor A performs surgery type 1 20% of the time and surgery type 2 80% of the time, so they're success rate is: (0.2×0.95)+(0.8×0.5)=0.19+0.4=0.59 or 59%.
- Doctor B has a 90% success rate for surgery type 1 and a 10% success rate for surgery type 2.
- Doctor B performs surgery type 1 80% of the time and surgery type 2 20% of the time, so they're success rate is: (0.8×0.9)+(0.2×0.1)=0.72+0.02=0.74 or 74%.

This is why it’s often useful to break down data into meaningful sub groups to get a more accurate understanding of relationships. We need to consider both the whole and the parts to get the full story.

## Conclusion

Probability allows us to handle uncertainty by quantifying the likelihood of different outcomes. In statistics, it underpins the process of making inferences, testing hypotheses, and predicting future events. For hands-on practice, I would reccommend using something like [brilliant.org](https://brilliant.org/) as they have plenty of examples to work through.