---
layout: post
title: "Trustworthy online experiments"
mathjax: true
---

Recently I read the book [Trustworthy Online Controlled Experiments: a practical guide to A/B testing](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264) by Kohavi, Tang, and Xu. It was promised in the title, but I was pleasantly surprised by just how practical it is! The authors have collectively worked at Microsoft, Amazon, Google, and LinkedIn, so they had a lot of great real life examples of both success and failure. While they don't go too deep on any one topic, they have a nice coverage of many topics that you usually won't find outside internal company guides. If you're just starting out with A/B testing in an online/app based company, this will give you a great structured introduction. However, even though I was familiar with the majority of the concepts discussed, I still got a lot out of it. In an effort to structure my thoughts, below I've summarised some of my learnings from the book with the addition of my own experience in some areas. Experimentation is a complex and ever evolving field, so it will be interesting to look back on this in coming years to see where better solutions have been found, or my opinion changes.


## Background

### Motivation

> One accurate measurement is worth more than a thousand expert opinions. <br>
> -- Admiral Grace Hopper

Imagine we launched a new feature and observed users who interacted with the feature. Maybe we see that users who interact with the feature are 50% less likely to churn than those who didn't. Does that mean our feature causes less churn? We don't know from this analysis because the users self selected into the the treatment, and are not necessarily similar to those who didn't. Perhaps more engaged users are more likely to interact with the feature, and thus they naturally have a lower churn rate (regardless of whether the feature is there or not). We can't say anything about the effect of the treatment on churn in this case. 

In it's simplest form, an experiment involves randomly splitting users into two groups: one sees no change from normal (control), and the other sees some new feature or change (variant). User interactions in both groups are monitored and logged (instrumented). Then at the end of the experiment, we use this data to assess the differences between the two groups on various metrics. By randomising which users are exposed to the feature in an experiment, we know that the only difference between the two groups is due to the change (plus some random variation). This is why experiments are the gold standard for establishing causality, and sit near the top of the [hierarchy of evidence](https://en.wikipedia.org/wiki/Hierarchy_of_evidence). When possible, a controlled experiment is often both the easiest and the most reliable and sensitive mechanism to evaluate changes. In addition to facilitating learning, experiments can also be great for rolling out features (as they can easily be rolled back if there are issues). Note, we still can't measure the effect of interacting with the feature, only the effect of being exposed, but this is generally enough to guide product development. I'll come back to this point later in more detail.


### Success in the wild

Tech companies like LinkedIn, Facebook, and Uber (just to name a few) run thousands of experiments a year. For example, testing user interface and product feature changes, relevance algorithms for search/ads/personalisation etc, performance and system improvements, and many more. It's very hard to guess which changes will be successful, and those that are successful are usually small incremental gains. However, by lowering the overhead, these companies have made it possible for almost anyone to test ideas. Every now and then, someone gets very lucky and a small change has a massive effect. For example, at Bing, adding some of the description to the headline of an ad, turned out to generate a $100M a year return. This idea sat in a backlog with low priority for 6 months before finally being picked up by an engineer, who spent only a few days on it. Similarly, despite a VP being dead against testing it, personalised recommendations based on a user's cart turned out to be very profitable for Amazon. They were also able to further improve on the original 'people who bought X bought Y', by generalising it to 'people who viewed X bought Y', and later 'people who searched X, bought Y'. Each time testing the switch in an experiment. Another area that's been shown again and again to have a positive effect on many metrics is performance improvements (e.g. speeding up page load can increase sales). 

However, don't get too excited, you shouldn't expect a large number of your experiments to succeed like this. It can also go the other way, for example Bing expected integrating with social networks would have a strong positive effect. After two years of experimenting and not seeing value, it was abandoned.

### Tenets for success

In chapter 3, three key tenets for success are discussed for organisations that wish to run experiments, paraphrased below:

- **1) They want to make data-driven decisions and have quantifiable goals.** There is a cost to experimentation and it WILL slow down progress. It's a lot easier to make a plan, execute on it, and declare success. However, this doesn't the change true effect. As I heard in a recent [interview](https://diggintravel.com/travel-digital-products-booking-com-case/), you might run faster blindfolded, but you're still going to hit the wall. It's also important to have quantifiable goals. It can be hard to find metrics that are measurable in the short term, sensitive to changes, and predictive of long term company goals;
- **2) They're willing to invest in infrastructure and tests to ensure the results are trustworthy.** Generally in tech companies it's relatively easy to get numbers, but hard to make sure they're trustworthy. In other industries, even being able to run experiments reliably may be difficult/illegal/unethical; and,
- **3) They recognise that they are not good at evaluating the value of ideas.** This is a hard one for some to accept, as team's build features and make changes because they think it will add value. However, based on the experience of many companies (including Microsoft, Google, Slack, and Netflix), you should expect 70-90% of your tests to fail to detect any effect. The more optimised the product becomes, the harder it gets to move the needle. This is just the harsh reality at least in customer-facing websites (though might differ in other domains). In response, most teams build iteratively to avoid investing too much time without signals of success (fail fast). For example, before building a complex model for some feature, test the idea with some simple business logic. 

Depending on the maturity level of the company, these might also develop over time. Chapter 4 of the book has some good advice on the stages of maturity and what to focus on if you're in the earlier stages.

## Part 1. Pre-experiment: experiment design

Though the standard A/B test sounds pretty simple at first, the design stage can actually be very involved in practice. Once I understand the hypothesis and treatment (e.g. what is the mechanism by which we expect to see an effect), below are some of the questions I'm thinking about at this stage:

- **What kind of set up do we need?** A/A (to get some data, or test our setup is working), A/B (one variant), A/B/n (multiple variants), or a holdout/blackout design.
![Experiment types](/assets/experiment_types.png)

- **What is a good control?** Sometimes a treatment combines a visual change with some logic, so it's important to isolate what is causing the effect. For example, let's say we add a block to the top of the page with a special offer. The block pushes down other content, which likely has an effect regardless of the content. In this case we might do: control (no block), variant 1 (block with just a header title or something neutral), variant 2 (block with the offer). If both variants are negative vs the control, the block is likely damaging and the offer isn't enough to offset it. This also comes up with testing models (e.g. adding an icon based on the prediction, in which case we might also want to test adding an icon based on simple logic or at random), and with ads (e.g. changing the layout might increase/decrease ads, which itself probably has a large effect regardless of the treatment).

- **What unique ID will be used for tracking (the randomisation unit)?** SUTVA (stable unit treatment value assumption) states the observations should be independent, so we want an ID that will avoid leakage or interactions between users. At an organisation level, you also need to decide what to do with abnormal users and bots (e.g. show base and exclude them, or still allocate at random but exclude them). Some common ID choices below and their pros/cons. Also keep in mind that all of your metrics should be done on the same level as the randomisation (e.g. if you randomise by cookie, your metrics will be something like average revenue per cookie).
![Tracking ID](/assets/tracking_ids.png)

- **How/where/when will visitors enter the experiment?** Proper randomisation is critical, and it's not as simple as it seems. The book goes into some depth on this. However, it's already good just to be aware that it's important, and look out for ways factors could influence the allocation. For example, when and where visitors enter the experiment can be a problem if they are in some way already exposed or influenced (e.g. tracking users who completed a purchase would not be a good choice for a change on the checkout page, as the change likely impacts the probability of purchasing). We also don't want to over-track visitors as this adds unnecessary noise, and can drag the average effect towards zero if the metric is not possible for these users, in both cases reducing power. For example, including all visitors when only those who log-in are exposed. Similarly, under-tracking is also an issue, as some observations are lost and if it's not at random the sample no longer represents the population well.

- **What size effect is meaningful and how certain do we need to be?** This is where the power analysis comes in. It's important that we control error rates by deciding the runtime upfront. Sometimes the required runtime is too long, in which case we need to re-evaluate and trade-off the confidence or size of the effect we can detect. More on how to do this [here](https://mverbakel.github.io/2021-04-11/power-calculations).

- **How safe is the experiment?** If we're unsure about the implementation or how users might react, we might want to ramp up the experiment, starting with a small proportion of traffic and slowly increasing to the planned split.

- **Do we need to be minful of other experiments?** If we have other changes we want to test that might interact, we could run them at the same time, splitting the traffic. Or, we could wait, and test them one by one. For example, if one experiment removes a feature while another changes the feature, there's an interaction and neither properly measures it's intended effect. This is particularly bad in visual changes that conflict (e.g. one experiment changes the background color while another changes the text color).

The above should then be wrapped up into a clear experiment plan and hypothesis. E.g. an example from the book is "Adding a coupon code field to the checkout page will degrade revenue per user for users who start the purchase process".

### Deciding Business Metrics (notes from chapter 6):

Metrics should be discussed at the team and company level to ensure there is alignment on the goals. The taxonomy used in the book is goals, drivers, and guardrails:

- **Goals** (north star): Based on a description of what the company ultimately cares about (why do you exist?), these are metrics that should capture progress towards success. They might be hard to move, requiring many initiatives, or a longer time period to observe change. It also might not be possible to perfectly capture the goal with metrics, so these might be proxies, and require iteration over time. However, these should be simple (easily understood and accepted) and stable (not updated every feature launch).
- **Drivers** (sign-post/surrogate /predictive metrics): Metrics that are thought to be drivers of success (goals), indicating you're moving in the right direction. For example, to increase revenue, we need to acquire customers, engage, and retain them. These should be aligned with goals, actionable and relevant, sensitive, and resistant to gaming (incentivise the right behaviours). Inspiration can be taken from the user funnel, or frameworks such as Google's [HEART framework](https://research.google/pubs/pub36299/): Happiness (attitudinal: satisfaction, willingness to recommend, perceived ease of use); Engagement (behavioral proxies like frequency, intensity, or depth of interaction over some time period, e.g. photos uploaded per user per day); Adoption (e.g. accounts created within 7 days); Retention (e.g. percentage of 7-day active users in a given week who are still 7-day active users 3 months later); and, Task Success (efficiency or time to complete a task, effectiveness or percent of tasks completed, and error rate). Another framework is AARRR!: Acquisition, Activation, Retention, Referral, and Revenue.
- **Guardrails** (protection metrics): Either protect the business or protect the trustworthiness of the experiment. For example, in a password management company security might be the goal, but ease of use and accessibility might be guardrails. Other common guardrails are performance metrics (e.g. load time, errors, crashes), costs like customer service tickets, and metrics that are not expected to change (e.g. pageviews per user). These could also be the driver or goal metrics of other teams (e.g. a team working on something like relevance might want to keep an eye on some form of revenue per user metric, ensuring it's sensitive enough). For a more holistic example, checkout how Airbnb has approached guardrails [here](https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669). Though [apparently](https://newsletter.bringthedonuts.com/p/building-products-at-airbnb) it's still a heavily debated topic internally. 
- **Diagnosis or debug metrics**: When the above metrics show a problem, these provide more granularity for drilling down. E.g. for click-through-rate, you might have the CTR of certain areas of the page, and for revenue you might have a binary indicator (1 if made any purchase), and the average revenue only for those where there is a purchase (i.e. the components of average revenue per visitor).  

Other taxonomies are briefly discussed: 1) assets (e.g. total Facebook accounts or connections) vs engagement (e.g. sessions or pageviews) metrics; and, 2) business (e.g. revenue per user, or daily active users, tracking business health) vs operational metrics (e.g. queries per second, tracking operational concerns).

#### Tips when developing business metrics:

- Less scalable methods like user research can be used to generate ideas, which can be validated with scalable analyses to get a precise definition. For example, we may see bounce rate (spending a very short time on the page) is correlated with dissatisfaction, then do an analysis of the distribution to find an exact cutoff that seems reasonable (e.g. 20 seconds).
- Consider measures of quality: If a user clicks but then clicks back shortly after, this might be considered a bad click. For LinkedIn, a good profile might be one with 'sufficient' information filled in. This can make driver metrics more interpretable.
-  If a model is used, it needs to be interpretable and regularly validated. For example, a popular metric is Lifetime Value (LTV), based on the predicted future revenue for a user. Netflix for example uses bucketed watch hours as it's easily understood and indicative of long-term retention.
- Sometimes it's easier to measure what you don't want. For example, a short visit after clicking a search result is usually more correlated with an unhappy user, whereas a long visit could mean they're happy (found what they want) or are unhappy and having a tough time finding what they want.
- Remember metrics are proxies, and the drivers and goals should be hard to game. For example, CTR might be used to measure engagement, but driving CTR alone could just increase clickbait. Similarly, short term revenue can be increased by increasing prices or ads, but these likely negatively harm customer retention. Additional metrics are needed to cover the edge cases, e.g. using human evaluation to measure relevance. Further, constrained metrics are less gameable e.g. ad revenue constrained by space on the page.
- Avoid vanity metrics and focus on user value and actions instead. For example, counting banner ads is not that useful (users can ignore), whereas counting 'likes' captures a user action (and is a fundamental part of the experience). 
- Be careful not to over pivot towards what is measurable, and in particular what is measurable in the short term. Other important factors like how customers feel about you product are very important, but difficult to measure. Taking into account multiple sources of information when making decisions can help mitigate this.

### Deciding Experiment Metrics (chapter 7)

> Measure a salesman by the orders he gets (output), not the calls he makes (activity)
> -- Andrew Grove

Not all business metrics will make good experiment metrics. Experiment metrics must be quantifiable, measurable in the short term (during the experiment), and sensitive to changes within a reasonable time period. For example, a high variance metric like revenue per user can be capped to improve power, whereas the original revenue data including extremes would be important for business reporting. Similarly, if your product is subscription based, the renewal frequency may be too long to see changes in an experiment, in which case surrogates like usage metrics may be used as early indicators.

In the book they talk about the concept of an Overall Evaluation Criterion (OEC), which (potentially) combining multiple metrics, gives a single metric that's believed to be causally related to the long-term business goals to optimise for. For example, for a search engine, the OEC could be a combination of usage (e.g. sessions per user), relevance (e.g. successful sessions, or time to success), and ad revenue metrics. It could also be a single metric like active days per user (if the target is to increase user engagement). To combine multiple metrics, they can first by normalised (0-1), then assigned a weight. This will require definition of how much each component should contribute. For example, how much churn is tolerable if engagement and revenue increase more than enough to compensate? If growth is a priority, there might be a low tolerance for churn. 

It's important we decide which metrics are important before the experiment, otherwise it can be tempting to build a story around whatever shows up as positive in the results (remembering if we test enough metrics we'll get some false positives). This is like the texas sharp shooter problem: it's easier to shoot first, then draw targets around the bullet holes. In statistics this is also called fishing or HARKing (hypothesising after the results are known).

### Metric examples 
Goal -> feature -> metrics

Amazon emails: Originally the team used click-through revenue, but this increases with email volume, leading to spamming. So while it optimised the short term revenue, users unsubscribed and then Amazon lost the chance to target them in the future. They later switched to an OEC with a lifetime loss estimate if the user unsubscribes. Even if the lifetime loss is estimated at only a few dollars, this showed more than half of their campaigns have a negative effect based on the OEC.

$$OEC = \frac{(\sum_i{revenue_i - unsubscriber * \text{lifetime loss}})}{n}$$

Bing: At Bing, progress is measured by query share and revenue. However, they once found a ranking bug which lowered the quality of results actually increased both as users were forced to do more queries and clicked more ads (results). This clearly doesn't align with Bing's long term goals. Using sessions per user would be better in this case as satisifed users will return and use Bing more. This example demonstrates what can happen when revenue metrics are used without constraints (we want to increase revenue, but not at the cost of other metrics like engagement). For ads, one way to enforce a constraint is to limit the average number of pixels that can be used over multiple queries.

New feature added: Click-throughs on the feature won't capture the impact on the rest of the page so could allow cannibalisation. A whole page click-through metric would be better (penalised for bounces), along with conversion rate (successful purchase), and time to success.

### Evaluating metrics:

- For new metrics, what additional information are they adding?
- For old metrics, periodically check if they've encouraged gaming (is there a notable behaviour of moving metrics just over the threshold), and whether there are any other areas for improvement.
- For both new and old, it's also important to regularly validate that the assumptions are met for the statistical tests being used. See my post [here](https://mverbakel.github.io/2021-04-03/metric-validation) on how to do this.
- Causal validation of the relationship between drivers and goals (difficult given correlation does not imply causation). Some methods for this include: checking if surveys / user research / careful observational analysis / external research mostly seem to point in the same direction; running experiments specifically to try test different parts; and, using a set of well understood past experiments to test new metrics.

### Improving sensitivity with variance reduction

One way we can improve the power of a test (sensitivity), is by reducing variance. For example, let's say you're at Amazon and you want to increase supply. You might run an experiment where you incentivise shops to add new products, and use the number of products per shop as a success metric. Before entering the experiment, some shops have only a few products, while others have thousands. Controlling for their pre-treatment number of products likely reduces the variance, increasing power and our ability to detect smaller effects. 

As demonstrated below, when variance is reduced, there's less overlap between the distributions of control and variant, making the effect more clear. 

![Variance reduction](/assets/high_low_variance.png)

Below are some ways we can achieve this:

- **Use a binary metric**: If still aligned well with the hypothesis, we could use a binary metric instead of a high variance continuous one. E.g. the number of searches has a higher variance than the number of searchers (1 if any search made, else 0). Similarly, Netflix uses a binary indicator of whether the user streamed more than X hours in a time period instead of measuring average streaming hours.
- **Cap/winsorize/transform**: If your metric is prone to outliers (unrelated to treatment), winsorizing (capping) values at some percentile will reduce variance (e.g. find the 1st and 99th percentile for the combined data, then cap values above/below those in both samples). 
- **Use stratification**: Divide the population into strata (e.g. based on platform, browser, day of week, country), then sample within each stratum separately, combining the results for the overall estimate. When sampling, we expect some random differences between the groups, so this is a way to reduce the differences (reducing variance). This is usually done at the time of sampling, but can also be done retrospectively during the analysis.
- **Use covariates**: Similar to the above, we can fit a regression controlling for data known pre-experiment. The variance from pre-experiment factors is unrelated to the treatment, so we can safely remove it without introducing bias. The Microsoft experiment team developed the method CUPED (Controlled-experiment using Pre-Experiment Data) for exactly this ([paper](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf), and an [article](https://booking.ai/how-booking-com-increases-the-power-of-online-experiments-with-cuped-995d186fff1d#:~:text=CUPED%20is%20a%20technique%20developed,power%20to%20detect%20small%20effects.) on how Booking.com uses this method. Ideally, the covariate is the same as the experiment metric but measured on the pre-experiment period (as the variance reduction is based on the correlation between the covariate and the metric).
- **Use a more granular unit of observation**: If possible, using a more granular ID will increase the sample size substantially (e.g. instead of user, you could use page views or searches). However, this is often difficult to do without any interaction or leakage (e.g. changing a page could have flow-on effects for the user's experience on other pages).
- **Paired experiment**: If it's possible to test both variant and control for the same user, a paired design will remove the between-user variability. For example, we can test two ranked lists at the same time with a design called interleaving (see this Netflix [post](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55) for examples).
- **Pooled control group**: If you have multiple experiments running at the same time, each with their own control group, you might be able to combine them to increase the sample size and power.


## Part 2. During the experiment: common issues

Instrumentation
Ramping exposure
Randomisation
Concurrent experiments


## Part 3. Post experiment: analysis

### Twyman's Law

> The more unusual or interesting the data, the more likely they are to have been the result of an error of one kind or another (Kohavi et al., chapter 3).

I've seen this time and time again. Errors are simply much more common than a truly large effect (e.g. logging issues, calculation errors, or missing/duplicated data). This is particularly a problem because we tend to create a story around positive results and want to believe it, but look to dismiss negative results and blame data problems etc. The advice here is to always treat those 'too good to be true' results with a healthy dose of skepticism as well.

### Statistics of A/B testing

In the A/B case, we have two independent random samples, and summary statistics for each of our metrics in each sample. Comparing the means for example, gives us an estimate of the average treatment effect in the population. However, by chance, we expect some random variation between the two groups, and so the observed difference will vary around the true effect in the population (i.e. even if the null is true and there is no efect, we will still observed differences). Therefore, it's important we correctly estimate the standard error (how much we expect estimates to vary with repeated sampling), so we can assess whether a difference is 'large enough' for us to reject the null hypothesis. To do this, we compare the observed difference to the range of values we would observe if the null was true (so assuming the null is true, how likely is it we would observe a difference at least as extreme as this). If the result is surprising, we reject the null and conclude there is likely an effect (the result is significant).

I've gone into some depth on the different tests common to A/B testing in previous posts. In general though, the t-test or Welch's t-test is most common for comparing the means, and a z-test for proportions or a g-test are common for comparing proportions. For binary metrics, see [here](https://mverbakel.github.io/2021-02-13/two-sample-proportions), and for continuous metrics see [here](https://mverbakel.github.io/2021-02-12/two-sample-ttests). 

Note, for ratio metrics (where the denominator is not the randomisation unit), and comparisons of the percentage change (also a ratio), the variance is not the same as that of an average. In these cases, the Delta method can be used, or alternatively bootstrapping. I talked more about this in my post on [confidence intervals](https://mverbakel.github.io/2021-03-05/confidence-intervals).

### Common mis-interpretations

- **Lack of power:** Often people will pick a standard runtime like 2 weeks without calculating the sample required for the effect size they're interested in. This can lead to under-powered experiments with little chance of detecting effects, even if meaningful effects exist. I think it's better to know this upfront, and trade-off your choices. More on this topic [here](https://mverbakel.github.io/2021-04-11/power-calculations).
- **p-values:** Related to the above, a non-significant p-value is often incorrectly interpreted as meaning there is no effect. In reality, we can't prove an absence of an effect, and it's also possible the effect was just too small to detect with this test. Another common mistake is to think it's the probability the null is true (again false). The p-value assumes the null is true, then tells us how often we would expect a result at least as extreme as that observed. If it's unlikely, we take that as evidence towards the alternate being true. A lot more on this topic [here](https://mverbakel.github.io/2021-02-20/pvalues).
- **Confidence intervals:** Similar to the p-value, confidence are also often misinterpreted. Though generally it's enough for stakeholders to understand it's a range of plausible values for the true population effect. More on this in this [post](https://mverbakel.github.io/2021-03-05/confidence-intervals).
- **Peeking at p-values:** It's not uncommon for people to monitor running experiments, including the p-value results. To control the error rates, we pre-determine the runtime, and it's important that we wait till the end of the runtime to make a decision. The p-value can fluctuate, so stopping early when you see it's significant will increase the False Positive Rate. If this is unsatisfying, you can also consider sequential testing, but it's a lot less common in practice.
- **Multiple testing:** When one test is inconclusive, it's tempting to try other metrics, definitions, or segments to see if there was any effect. However, the more tests we run, the higher our false positive rate becomes. For example, if the FPR per test is 5% and we run 20 tests, the probability of at least one False Positive become 64.2% (see this [post](https://mverbakel.github.io/2021-02-20/pvalues) for the workings). One approach for experiments is to separate metrics into 3 groups with descending significance levels based on priority (e.g. primary can be 0.05, secondary 0.01, but third order should 0.001). The logic is based on how much we believe the null is true pre experiment. The stronger the belief, the lower the significance level you should use.

### Analysing segments

Given the effect of an experiment is not always uniform, it can be useful to break down the results for different segments. For example: market or country (e.g. could be different reactions by region, or localisation issues), device or platform (e.g. web/mobile/app, iOS/Android, and browser breakdowns can help with spotting bugs), time of day and day of week (plotting effects over time), user characteristics (did they join recently e.g. within last month, account type (single/shared, business/personal)). However, avoid comparing metrics for sub-segments that were impacted by the treatment. As users have shifted between the sub-segments, the differences observed are not necessarily only due to the treatment. For example, if your treatment increases bookings, comparing those who do and don't cancel will probably show different proportions of control/variant than expected.

Differences in the effect by segment are called heterogeneous treatments effects (HTE), meaning the effect is not the same for all observations. Always think about whether the sub-segments are comparable though, as sometimes the differences may not be due to the treatment (e.g. the click through rate can differ for different operating systems because of different tracking technologies). If you are doing many breakdowns, also remember to correct for multiple testing. Alternatievly, one way to deal with this is to do many breakdowns (which could also be done with ML, e.g. a decision tree), identify and explore those that are interesting, and then validate them in a new experiment or with other methods on new data.

**Simpson's paradox**

Simpson's paradox is when we find different directions of association between small groups and the overall totals. It happens when we compare a treatment across two groups, and the proportions are not equal (e.g. doctor A treats more hard cases than doctor B, so even though doctor A has a higher success rate for both hard and easy cases, when the data is aggregated doctor B appears more successful). We can see this in experiments when ramping is used. For example, if we start with 1% in variant, then the next day increase to 50% in variant, even if the CR is higher in variant on both days, the table below shows how we end up with an overall negative effect (as day 1 counts more towards the control total). The problem is a higher proportion of the treatment observations are in day 2, and the conversion rate was in general lower on day 2.

![Simpsons paradox with ramping](/assets/simpsons_paradox.png)
Example from: [Kohavi et al., chapter 3](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264)

This can also happen when the sampling ratio varies by country, or some other variable (e.g. browser, or customer value segment) and the results are then combined. The message here is that you should be careful when aggregating data collected at different percentages.

### Making a decision

Our ultimate goal is to use the data we've collected to make a decision, to ship the change, or not to ship. This requires some context, and often draws on other sources of information as well. 

**Considerations:**
- Metric trade-offs: The direction of the effect can differ across metrics, meaning we need to make some trade-offs. E.g. if engagement metrics improve, but revenue decreases, should we ship? The answer will depend on the goals of the test, and by how much each changes.
- Do the benefits outweigh the costs: We need to consider the cost of fully building out the feature (if not yet complete, e.g. MVP), and the potential maintenance costs (e.g. are we adding complexity that could cause more bugs in the future, or make it harder to change). 
- Downsides of making a wrong decison: For example, if the change is temporary (like a limited time offer), then the cost of making a mistake is limited and we might set low bars for statistical and practical significance. 

**Potential outcomes:**
- Result is not statistically or practically significant: Iterate or abandon the idea. If the bounds are very wide, you likely don't have enough power to draw a strong conclusion. Consider running a follow-up experiment with more power.
- The result is practically significant, but not statistically significant (the change could be meaningful, or not exist at all). Like in the previous case, we would need a follow-up experiment with more power.
- Result is statistically and practically significant: Launch.
- Result is statistically significant, but not practically significant: May not be worth launching, consider costs. If the bounds are wide (could be practically significant, could not be), then like before we can retest with more power.


### Long-term effects



### Threats to experiment validity

We often talk about two main types of validity for experiments: internal and external.

**1) Internal validity (truth in the experiment):**
Does it correctly measure what we intended to measure? This is mostly about how well the experiment is designed and managed. In general, experiments have high internal validity compared to observational analyses because the design controls for all the potential confounders by randomising the treatment (assuming everything is set up well). However, some common issues that arise are: 
- Stable Unit Treatment Value Assumption (SUTVA) violation. For most tests we assume the observations are sampled independently: they don't influence each other, each is counted only once, and each is only given one treatment. This can be violated for example if there's a relationship between customers (e.g. in social networks a feature might spillover to their network), the ID is not unique (e.g. http cookies can be refreshed), or the supply is in some way shared between the control and variant (e.g. if you lower the price in variant, this likely lowers supply in control).
- Survivorship bias: This is about capturing all observations. In the book they give an example about the military looking at where returning planes had the most damage, with the intention of understanding where to add armor. Given the bullets were relatively uniform, it would actually be best to add armor to the places not damaged (as the planes damaged in those locations didn't make it back).
- Self-selection. Given we can't control whether users actually interact with a treatment (e.g. we can send an email, but we can't make someone open it), it might be tempting to only analyse those who actually interacted. However, this introduces selection bias, so instead we measure the effect of the intention to treat (e.g. being sent an email).
- Sample Ratio Mismatch. Sometimes due to errors in the design or setup, the ratio of users in control and variant doesn't match the intended split (e.g. bots, browser redirects, and lossy instrumentation could be to blame). Even a small imbalance can completely invalidate the experiment, but don't just eyeball, use a test like demonstrated in this [SRM checker](https://www.lukasvermeer.nl/srm/docs/faq/). If there's an SRM, you'll need to investigate, correct, and restart.
- Residual effects. Often bugs are identified when an experiment is first launched, and the experiment is continued after fixing. Be wary that those users could still return and be impacted by their earlier experience. Re-randomising redistributes the effect, but it might be best to leave some buffer between the experiments.
 
**2) External validity (truth in the population):**
Can the results be generalised to new people/situations/time periods beyond what was originally tested? Some of the key threats to external validity include:
- Primacy effects: users are 'primed' on the old feature, so it may take time for them to adjust to the new feature (run for longer).
- Novelty effects: the inverse of primacy, users are initially intrigued by the new feature, but the effects wear off over time.
- To quick ways to detect primacy and novelty effects is to plot usage over time, as well as the effect over time based on when users joined. We generally assume the effect is constant over time, so if it's increasing/decreasing we may need to run longer to see if it stabilises.
- Seasonality: if you run the experiment during an 'unusual' time (e.g. Christmas), the observed effects may not generalise to other time periods. For example, at Bing, purchase intent rises dramatically around December, and so ad space is increased, and revenue (per search) increases. It's generally advised to avoid testing new features from around mid December until after the new year for this reason.
- Hour/day seasonality or variations: You generally want to run an experiment for at least a week as users can behave very differently depending on the day of week.
- Population changes combined with heterogeneous treatment effects: for example, if the effect differs by country and the mix of countries changes, this would change the average treatment effect. Though, if you want to test on a completely new country/product/etc., it's usually easy enough to just re-run the experiment. 
- Changes in the effect of a treatment: e.g. the requirements and behaviours of customers can change drastically due to an external shock like a pandemic. 
- This is a very relevant topic right now because the pandemic has changed people's lives and behaviours so much, meaning we have both composition changes (combined with heterogeneous treatment effects), and changes in the actual effect of treatments. For example, in the travel industry, we see some countries are more impacted than others, and the effects of treatments often differ by country, so the average treatment effect observed will likely change simply because of the change in the mix of customers. Further, there's been many changes in the preferences of customers and what influences they're decisions (e.g. more demand for local and car based travel, and a greater need for flexibility with cancellations or date changes). Therefore, it seems likely that the effect of treatments may differ in the future 'normal'. 
- If you have a stable ID, a good solution might be a long term holdout (leaving it for months to assess the longer term effects). 

**3) Statistical validity:**
It's also worth mentioning that statistical validity is also a common issue in companies where anyone can run an experiment. It's important that appropriate statistical tests are chosen (usually built-in to the company tool), but also that the assumptions of the test are met (see my [post](https://mverbakel.github.io/2021-04-03/metric-validation) for how to validate continuous metrics). However, often the most common mistakes are simple, e.g. not adhering to the runtime and/or stopping when significance is reached.


## Other common topics

## Opt-in products

Delayed Activation (e.g. opt-in products): Sometimes there can be a bias in experiments due to product activation bias. In this case we can randomise assignments for some recruitment period, and then continue to observe for some observation window to give time for people to activate. 

## Experiments with an expected impact on performance

Sometimes we expect a new feature will negatively impact performance (e.g. calculations are required before showing the page, slowing down the page load). As mentioned earlier, speed matters a lot. Before we spend time optimising our feature to address the performance impact though, we want to know if it has any value. One way to separate the effect of the performance from the effect of the feature is to add a variant that does the calculations but doesn't show the change:

- 1) control (no change); 
- 2) variant 1 (do the calculations etc., but then don't show the change), and 
- 3) variant 2 (do the calculations etc., and show the change). 

The difference between the control and variant 1 is the effect of any performance loss (as the only difference is the calculations etc. being done in variant 1). The difference between variant 1 and variant 2 gives us the effect of the feature (assuming no performance impact), as both should have the same performance. Finally, the difference between variant 2 and the control is the total effect of switching to this feature (including the effect of any performance change). If this total effect is positive, any performance loss is outweighed by the effect of the feature and so we could ship the change. However, if the total effect is negative, but the feature effect is positive and the performance effect is negative, we'll need to work on optimisations to address the performance issues. 

## Block designs
Deliveroo example.

## How to drive strategy with experiments?

If a business strategy can be well defined by an OEC, experimentation can provide a great feedback loop on whether it's working, and help the company hill-climb to a local optimum. For example, testing minimum viable products (MVPs) in experiments can help to identify areas which are most promising for the OEC (quickly narrowing down where to focus before committing significant resources). A well defined OEC can also accelerate progress as teams are empowered to optimise and improve on it. However, this is also why it's important that it's aligned well with the goals and is not gameable, as that may incentivise the wrong outcomes (e.g. trying to improve the lighting on a sinking ship, instead of prioritising passenger safety). Similarly, if crashes are increasing, the experience is so bad that any other factors should be considered second. Instead of trying to capture everything in the OEC, another option is to add guardrail metrics for things that the organisation is not willing to change (e.g. crash rates).

Instead of hill-climbing, the company may also want to try something far from the current (e.g. a total site redesign). These are more likely to fail, but with bigger risk also comes a bigger potential pay-off (jumping to a bigger hill). However, this can be a bit more difficult to test with experiments:

- Longer and larger experiments may be needed: big changes can suffer from primacy effects (change aversion), but should eventually stabilise to the longer terme effect. 
- May need many experiments: as each experiment is testing one idea/tactic. Strategies are broad and one tactic failing doesn't mean the strategy is bad, but if many fail, it might be time to rethink. For example, after spending ~$25M and 2 years without a significant impact on key metrics, Bing had to give up on their plans to integrate with social media platforms. It may be hard to give up on a big bet, but these are sunk costs, so we should make forward-looking decisions based on the data and accept these cases as "achieved failures" (coined by Eric Ries).

Some argue experimentation is too slow to inform strategy, but with MVPs and iterative testing it's still a very useful way of reducing the uncertainty. However, it's also important to consider the Expected Value of Information (EVI), i.e. how will additional information help decision making? 

## When not to experiment?

This leads us to the question of when not to experiment.

We at Stack Overflow have encountered this situation when considering a feature used by a small number of users and a potential change to that feature that we have other reasons for preferring to the status quo. The length of a test needed to achieve adequate statistical power in such a situation is impractically long, and the best choice for us in our real-life situation is to forgo a test and make the decision based on non-statistical considerations.

“Product thinking is critical here. Sometimes a change is obviously better UX but the test would take months to be statistically significant. If we are confident that the change aligns with our product strategy and creates a better experience for users, we may forgo an A/B test. In these cases, we may take qualitative approaches to validate ideas such as running usability tests or user interviews to get feedback from users,” says Des. “It’s a judgement call. If A/B tests aren’t practical for a given situation, we’ll use another tool in the toolbox to make progress. Our goal is continuous improvement of the product. In many cases, A/B testing is just one part of our approach to validating a change. ”


## Complementary / alternative methods

Quasi-experiment designs: geo, randomising time...



## Estimating the cumulative impact of experiments

We can’t just add up the value of each fullon: 
We’ll overestimate our impact
Experiments use different metrics
We hope they all influence the primary metric at the end of the project

Need to use a holdout or blackout.


Type S: sign error (probability estimate being in the wrong direction)
Type M: magnitude error (factor by which magnitude might be overestimated).


Bing Relevance example (chapter 1). Hundreds of people are all working to improve a single OEC by 2% per year, which is calculated as the sum of the treatment effect from all shipped (full-on) experiments (assuming they are independent, so the effects are additive). Given they run thousands of experiments and some may appear positive by chance, after a successful experiment (possibly after many iterations to refine it), a 'certification' experiment is run with a single treatment (variant). The result of this replication experiment is what is used as credit towards the 2% target.


## Conclusions

I'll finish with some sage advice from the book that resonated with me.

> Good data scientists are skeptics: they look at anomolies, they question results, and they invoke Twyman's law when the results look to good. ([Kohavi et al,. Chapter 3](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264))
