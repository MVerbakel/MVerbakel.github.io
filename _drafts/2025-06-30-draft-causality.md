---
layout: post
title: "Causal inference"
mathjax: true
---

Causal inference spans multiple disciplines, and is therefore a very broad topic. However, my interest is primarily in applying it to real world problems in industry as a Data Scientist. With that context, I've focused on the following topics: the motivation, DAGs, estimating ATE, estimating CATE, and validation. I first want to start with my own motivation for learning more about causal inference. As a data scientist working in industry, we're faced with a lot of causal questions (e.g. what drives conversion? who is most likely to respond to a promotion? what happens if we increase prices?). With the exception of randomised experiments, it's very common for people to rely on correlations from observational data to answer questions like these. Even with the caveat that correlation doesn't imply causation, you can come up with a story that makes any result feel convincing. This isn't always harmful, and not all decisions should rely on data. However, when we do need data to inform a decision, I think it's important to focus on causal relationships. This take a lot of thinking, as well as subject matter expertise, but I would argue that it saves time in the long run as you're less likely to chase ghosts. With AI tools speeding up other tasks, investing your brain power here can add a lot of value.



Largely focused on the Potential Outcomes framework (Rubin), as this aligns well with experimentation and estimation in industry. I haven't dived into do-calculus much (Pearl's framework), but really like DAGs for study design and clarifying assumptions.



One common mistake is ignoring confounders — variables that influence both the treatment and outcome. For example, while hospitalization is correlated with death, it’s actually the underlying illness (a confounder) driving both, not the hospital itself causing harm.

Another error is over-correcting by adjusting for colliders — variables influenced by both the cause and effect. For instance, studying only hospitalized patients might reveal a spurious link between COVID-19 and diabetes, not because one causes the other, but because hospitalization is affected by both.

A third pitfall is adjusting for mediators, which lie on the causal path. If we control for conditions like lung cancer when studying the effect of smoking on mortality, we may mistakenly conclude that smoking has no impact, even though it clearly does through those mediators.

The core challenge of causal inference is that we rarely have perfect data: confounders may be unobserved, and distinguishing between confounders, mediators, and colliders is not always straightforward. In complex systems, causal relationships may even run in both directions, making clean interpretations especially difficult.



You’ve probably heard the phrase “correlation is not causation.” A treatment has a causal effect on an individual's outcome if their outcome with the treatment differs from what it would have been without it. The catch is that we can never observe both outcomes for the same individual at the same time. Take a simple example: if you send a customer an email, you can observe whether they return after, but you’ll never know for certain what they would have done without the email. This is where the potential outcomes framework comes in. The observed outcome is called the factual, and the unobserved one is the counterfactual. While we can’t observe both for any one individual, we can estimate average effects across groups if the treated and untreated groups are otherwise similar. Having a well defined causal question requires being explicit about the intervention of interest, the relevant data, and variables that need to be adjusted for (e.g. confounders). 

Randomized experiments are a gold standard because we control the assignment of treatment, ensuring that any difference in outcomes is due to the treatment itself. However, experiments aren’t always feasible, so there are many situations where we need to rely on observational data to estimate causal effects. Treatments can be active interventions (like sending an email), but also events we observe (like an external policy change or app outages). This is where the distinction between association and causation becomes critical. Just because two variables move together in the data doesn't mean one causes the other. For example, ice cream sales and beer sales often rise together in the summer, but both are caused by a third variable: temperature. To identify true causal effects using observational data, we need to think carefully about the mechanism generating the data. If we casually fit a model to the data, we can’t tell whether X causes Y, Y causes X, or whether both are influenced by some third variable Z. When experiments aren't possible, it helps to start by imagining the experiment you wish you could run (target trial) and then try to approximate it using observational data. Uncertainty is unavoidable with observational data, and often the best you can do is be explicit about your assumptions and use sensitivity tests to try quantify the potential impact of bias. This allows others to challenge them in a more productive way. Ultimately, causal discovery from observational data relies heavily on subject matter expertise and the plausability of the proposed causal graph.  

Sometimes this takes a passive approach, meaning we don't involve ourselves in the data generation, but merely want to evaluate the average causal effect of some existing treatment (e.g. was a policy change effective). However, in business we typically have levers to pull to achieve our goal, and so we can actively change a customer's experience to estimate the effect for different segments, allowing personalisation. 

Key points: Target trial, causal graph, systematic bias, random variability, models
Could do a separate post on applying this to industry?

---
# Notation

Let's start with some mathematical notation and definitions. This will make it easier to follow the explanations, but you can refer back as you go. Note, capital letters are used for random variables, and lower case for particular values of variables.

| **Notation**                  | **Definition**                                                                                                                                                               |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $$T_i$$                     | **Treatment** indicator for observation $$i$$. Takes the value 1 if treated, 0 if not treated. A treatment refers to any condition, event, action, or exposure that might influence an outcome.                                                                                 |
| $$Y_i$$                     | The **observed outcome** for observation $$i$$.                                                                                                                                |
| $$Y_{\text{0i}}$$           | The **potential outcome** for observation $$i$$ without treatment (i.e., counterfactual outcome under control).                                                               |
| $$Y_{\text{1i}}$$           | The **potential outcome** for observation $$i$$ with treatment (i.e., counterfactual outcome under treatment).                                                                 |
| $$Y_{\text{1i}} - Y_{\text{0i}}$$ | The **individual treatment effect** for observation $$i$$. We generally cannot know this because we only observe one of the potential outcomes (either $$Y_{\text{1i}}$$ or $$Y_{\text{0i}}$$). |
| $$ATE = E[Y_{\text{1}} - Y_{\text{0}}]$$ | The **average treatment effect** (ATE), which is the expected value of the treatment effect across a population of individuals.                                                  |
| $$ATT = E[Y_{\text{1}} - Y_{\text{0}} \mid T=1]$$ | The **average treatment effect on the treated** (ATT), which is the expected treatment effect among those who actually received treatment ($$T=1$$).                               |
| $$\perp\!\!\!\perp$$         | Indicates **independence**. For example, $$A \perp\!\!\!\perp B$$ indicates that $$A$$ and $$B$$ are independent.                                                                   |

If $$T_i = t$$, then $$Y_\text{ti}=Y_i$$. That is, if observation i was actually treated (t=1), their potential outcome under treatment Y is equal to their observed outcome.

---
# Background

## Individual effect

A treatment has a causal effect on an individual's outcome if their outcome with the treatment differs to their outcome without it. To estimate this, we need an outcome, a treatment, and the potential outcomes for an individual. Let's pretend we have two identical planets, and the only difference is that on Earth 1 no one plays tennis ($$Y^\text{t=0}$$) and on Earth 2 everyone plays tennis ($$Y^\text{t=1}$$). The outcome is whether you get cancer, and the treatment is playing tennis. The table below shows there is a causal effect of tennis on cancer for Alice, Bob and Eva because the difference in the potential outcomes is non-zero. Alice and Eva get cancer if they don't play tennis, and don't get cancer if they do. The inverse is true for Bob. 

| **Name**    | **$$Y^\text{t=0}$$** | **$$Y^\text{t=1}$$** | **Difference** |
|-------------|-----------------|------------------|----------------|
| Alice       | 1               | 0                | -1             |
| Bob         | 0               | 1                | 1              |
| Charlie     | 1               | 1                | 0              |
| David       | 0               | 0                | 0              |
| Eva         | 1               | 0                | -1             |
| Fred        | 0               | 0                | 0              |
|-------------|-----------------|------------------|----------------|
| Average     | 3/6             | 2/6              |-1/6            |

## Average effect

If we take the average treatment effect in the example above ($$ATE=-\frac{1}{6}$$), we would conclude tennis decreases the chance of you getting cancer. We can also see this in the average cancer rate for those with and without treatment (as the average of the differences is equal to the difference of the averages). With tennis ($$Y^\text{t=1}$$) they had a $$\frac{2}{6}$$ chance, whereas without it ($$Y^\text{t=0}$$) they would have had a $$\frac{3}{6}$$ chance ($$E[Y^\text{t}] \neq E[Y^\text{t'}]$$). Keep in mind there can be individual causal effects even if there is no average causal effect. For example, Alice has an individual causal effect because she avoids cancer if she plays tennis, even if the average showed no difference. While the average causal effect is most commonly used, you can also use other differences such as the median or variance.

## Effect measures

There are several ways to represent the comparison between $$E[Y_t]$$ and $$E[Y_{t'}]$$, each corresponding to a different scale. For a binary outcome, these effect measures can be written as:

1. **Risk Difference**: The absolute difference in probabilities. It reflects the average change in risk due to the treatment and is equal to the average of the individual treatment effects. If the risk difference is zero, there is no effect of treatment.

   $$\text{Pr}(Y_t = 1) - \text{Pr}(Y_{t'} = 1)$$

2. **Risk Ratio**: Tells us how many times more likely the outcome is in the treatment group compared to the control group. The risk ratio is not the same as the average of the individual ratios, as it is non-linear.

   $$\frac{\text{Pr}(Y_t = 1)}{\text{Pr}(Y_{t'} = 1)}$$

3. **Odds Ratio**: Compares the odds of the outcome occurring in the treatment group relative to the control group. Like the risk ratio, it is also non-linear and does not represent the average of individual causal effects.

   $$\frac{\text{Pr}(Y_t = 1)/1 - \text{Pr}(Y_t = 1)}{\text{Pr}(Y_{t'} = 1)/1 - \text{Pr}(Y_{t'} = 1)}$$

The choice between these depends on the nature of the outcome, the context of the study, and the specific goals of the analysis. If your goal is to understand absolute effects, the risk difference might be the best choice. If you're interested in understanding relative effects, or the multiplicative effect of treatment, the risk ratio or odds ratio might be more appropriate. Odds tend to behave more stably than probabilities, especially when the event is rare. This makes the odds ratio less sensitive to small changes in event rates in rare-event situations.

## The fundamental problem

The fundamental problem of causal inference is that we can never observe the same person/observation with and without treatment at the same point in time. Using another example, if you send a customer an email to try improve retention, you don't know what they're retention would have been without it. While we might not be able to measure individual effects, we can sometimes measure average effects. This is where potential outcomes comes in: the actual observed outcome is factual and the unobserved outcome is the couterfactual. If we compare those with and without treatment and they are otherwise similar, we can estimate the average effect. 

| **Name**    | **$$Y^\text{t=0}$$** |**$$Y^\text{t=1}$$** |
|-------------|-----------------|------------------|
| Alice       | 1               | ?                |
| Bob         | ?               | 1                |
| Charlie     | 1               | ?                |
| David       | ?               | 0                |
| Eva         | ?               | 0                |
| Fred        | 0               | ?                |
|-------------|-----------------|------------------|
| Average     | 2/3             | 1/3              |

$$ATE = E[Y_\text{1} - Y_\text{0}] = 1/3 - 2/3 = -1/3$$

Now in our two Earth's example, if we randomly selected the samples, this would be an unbiased estimate of the average causal effect. However, if we just took a sample of people from our one Earth and compared those who played tennis to those who don't, we run into the problem of bias because all else is not equal (there are other factors influencing the cancer rate that also impact the likelihood of playing tennis, i.e. confounders). We can say from this there's an association, but not necessarily causation.

## Understanding bias

Bias is the key concept that separates association from causation. Intuitively, when we compare observational data, we will consider why a difference might exist that isn't due to the treatment itself. For instance, playing tennis requires time and money, which means tennis players may also have higher incomes and engage in other health-promoting behaviors, such as regular doctor visits. This could explain some or all of the observed difference in cancer rates.

We cannot observe the counterfactual outcome, $$Y_0$$, for the tennis group. In other words, we can’t know what their cancer risk would be if they didn't play tennis. However, we can reason that $$Y_0 \mid T=1$$ (the cancer risk of tennis players if they didn’t play tennis) is likely lower than $$Y_0 \mid T=0$$ (the cancer risk of non-tennis players), implying that the basline differs (it's not a fair comparison). Using the potential outcomes framework we can express this mathematically. An association is measured as the difference in observed outcomes:

$$E[Y \mid T=1] - E[Y \mid T=0] = E[Y_1 \mid T=1] - E[Y_0 \mid T=0]$$

We can separate the treatment effect (ATT) and the bias components by first adding and subtracting the counterfactual outcome $$E[Y_0 \mid T=1]$$ (the outcome for treated individuals had they not received the treatment): 

$$E[Y \mid T=1] - E[Y \mid T=0] = E[Y_1 \mid T=1] - E[Y_0 \mid T=0] + E[Y_0 \mid T=1] - E[Y_0 \mid T=1]$$

Then re-arranging the equation:

$$= E[Y_1 - Y_0 \mid T=1] + E[Y_0 \mid T=1] - E[Y_0 \mid T=0]$$

Where:
- $$E[Y_1 - Y_0 \mid T=1]$$ represents the Average Treatment Effect on the Treated (ATT), i.e. the treatment effect for those who received the treatment (tennis players).
- $$E[Y_0 \mid T=1] - E[Y_0 \mid T=0]$$ represents the bias, which is the difference between the potential outcomes of the treated and untreated groups before the treatment was applied. This reflects pre-existing differences (e.g. higher income, healthier lifestyle) that can affect the outcome.

In our tennis example, we suspect that $$E[Y_0 \mid T=1] < E[Y_0 \mid T=0]$$, meaning that tennis players, even without tennis, may have a lower cancer risk than non-tennis players. This bias will likely underestimate the true causal effect of tennis on cancer rates.

## Addressing the bias

To obtain an unbiased estimate of the treatment effect, we need the treated and untreated groups to be exchangeable before the treatment is applied ($$E[Y_0 \mid T=1] = E[Y_0 \mid T=0]$$, so the bias term becomes 0). In this case, the difference in means can give us a valid causal estimate for the Average Treatment Effect on the Treated (ATT):

$$ATT = E[Y_1 - Y_0 \mid T=1]$$ 

$$= E[Y_1 \mid T=1] - E[Y_0 \mid T=1]$$

$$= E[Y_1 \mid T=1] - E[Y_0 \mid T=0]$$

$$= E[Y \mid T=1] - E[Y \mid T=0]$$

If the treated and untreated groups are expected to respond to the treatment similarly (exchangeable), then we can assume that the treatment effect is the same for both groups. In that case:

$$E[Y_1 - Y_0 \mid T=1] = E[Y_1 - Y_0 \mid T=0]$$

$$ATT = E[Y_1 - Y_0 \mid T=1] = ATE$$ 

This is the goal of causal inference: to remove bias so we are left with a consistent estimate of the Average Treatment Effect (ATE).

## Randomised Control Trials (RCT)

In an RCT there will still be missing values for the counterfactual outcomes, but they occur by chance due to the random assignment of treatment (missing at random). Imagine flipping a coin for each person in your sample: if it’s heads, they get the treatment; if it’s tails, they get the control. Since the assignment is random, the two groups (treatment and control) should be similar on average, making them exchangeable (i.e., the treatment effect would have been the same if the assignments were flipped). With a fair coin it's a 50/50 chance of treatment, but this is not essential, it could also be 80/20 or some other ratio.

The potential outcomes in this case are assumed to be independent of the treatment (i.e. treatment assignment doesn’t influence the potential outcomes), which we express mathematically as: $$ Y_t \perp\!\!\!\perp T $$. You could check this by comparing pre-treatment values of the two groups to make sure there's no significant difference. Given the expected outcome under control ($$E[Y_0 \mid T=0]$$), is equal to the expected outcome under control for those who received treatment ($$E[Y_0 \mid T=1]$$), any difference between the groups after treatment can be attributed to the treatment itself ($$ E[Y \mid T=1] - E[Y \mid T=0] = E[Y_1 - Y_0] = ATE $$). It is important not to confuse this with the statement: $$ Y \perp\!\!\!\perp T $$ which would imply that the outcome is independent of the treatment, meaning there would be no effect of the treatment on the actual outcome.

RCTs are considered the gold standard in causal inference because randomisation eliminates many sources of bias. However, they are not always feasible or ethical to conduct, especially when the treatment itself could have harmful effects or when random assignment would be impractical in certain settings. For example, a common issue in online experiments is wanting to know the long term effect, but only being able to measure the short term effect (e.g. 2 weeks) due to both tracking (e.g. stability of the randomisation unit) and business reasons (e.g. needing to make a decision).

## Cross over experiments

One way we could measure both the potential outcomes for an individual in an experiment is to observe the same individuals at different time periods (e.g. give treatment at t=0 and control and t=1). However, this requires: no carryover effect (abrupt onset which completely resolves by the next time period); the effect doesn't depend on time; and, the counterfactual under no treatment doesn't depend on time. It's often hard to argue these are true in real life.

## Conditional randomisation & standardisation

The randomised experiment covered above was an example of a marginally randomised experiment because it had a single unconditional (marginal) randomisation probability for all participants (50% chance of treatment). Another type of randomised experiment is a conditionally randomised experiment, where we have multiple randomisation probabilities that depend on some variable (S). For example, in the medical world we may apply a different probability of treatment depending on the severity of the illness. We flip one (unfair) coin for non-severe cases with a 30% probability of heads (treatment), and another for severe cases with a 70% probability of heads (treatment). Now we can't just compare the treated to the untreated because they have different proportions of severe cases. However, they are conditionally exchangeable: $$ Y_t \perp\!\!\!\perp T \mid S=s$$. This is to say that within each subset (severity group), the treated and untreated are exchangeable. It's essentially 2 marginally randomised experiments, within each we can calculate the causal effect. It's possible the effect differs in each subset (stratum), in which case there's heterogeneity (i.e. the effect differs across different levels of S). However, we may still need to calculate the population ATE (e.g. maybe we won't have the variable S for future cases). We do this by taking the weighted average of the stratum effects, where the weight is the proportion of the population in the subset (this is called standardisation):

$$Pr[Y_t=1] = Pr[Y_t = 1 \mid S =0]Pr[S = 0] + Pr[Y_t = 1 \mid S = 1]Pr[S = 1]$$

$$\text{Risk ratio} = \frac{\sum_\text{s} Pr[Y = 1 \mid S=s, T=1]Pr[S=s]}{\sum_\text{s} Pr[Y = 1 \mid S=s, T=0]Pr[S=s]}$$

Conditional exchangeability is also referred to as ignorable treatment assignment, no omitted variable bias or exogeneity.

## Inverse Probability Weighting (IPW)

Mathematically, this is the same as the process of standardisation described above. Each individual is weighted by the inverse of the conditional probability of treatment that they received. Thus their weight depends on their treatment value T and the covariate S (as above, which determined their probability of treatment): if they are treated, then the weight is $$1 / Pr[T = 1 \mid S = s]$$; and, if they are untreated it's: $$1 / Pr[T = 0 \mid S = s]$$. To combine these to cover all individuals we use the probability density function instead of the probability of treatment (T): $$W^T = 1 / f[A \mid L]$$. Both IPW and standardisation are controlling for S by simulating what would have happened if we didn't use S to determine the probability of treatment.

---
# Observational data

The issue with observational data is that treatment is not randomly assigned. To approximate the causal effect in this situation, one option is to treat it as a conditionally randomised experiment, i.e. we assume treatment was randomly assigned conditional on some set of measurable covariates (S). This requires the causal effect to be identifiable, meaning all 3 of these conditions hold:

- **Consistency**: Treatment values correspond to well defined interventions, and thus well defined counterfactual outcomes (i.e. there is a precise definition of the counterfactual outcomes that can be measured by the observed outcomes). This is based on judgement, but involving domain experts can help you assess if it's precise enough (e.g. "weight loss" is too vague as there are many ways this can be implemented (exercise, diet, drugs) with very different outcomes);
- **Exchangeability**: The conditional probability of treatment depends only on the covariates (S);
- **Positivity**: The probability of treatment conditional on S is positive (i.e. some individuals will be assigned to treatment and some to control). 

In fields like epidemiology this can be justifiable. However, in other cases such as business and economics it's often hard to argue conditional exchangeability, which is not gauranteed without random assignment. Recall, conditional exchangeability holds in the experiment example above because within each level of the covariate S (severity level), all other outcome predictors are equally distributed between the control and treated groups due to random assignment ($$ Y_t \perp\!\!\!\perp T \mid S=s$$). That example also could have been an observational study, where doctors used the severity of the illness to decide the treatment assignment, as long as there is still a positive probability of both treatment and control within each severity group (i.e. sever cases are not all given treatment). If that's the only outcome predictor (confounder) that is unequally distributed between the control and treatment group, then we could use standardisation or IPW to estimate the causal effect like before. However, there are often multiple reasons, and therefore multiple covariates that are unequally distributed. For example, perhaps there are other unmeasured predictors (U) like smoking status that doctors considered. We only care about the one's which are outcome predictors (i.e. confounders). However, no matter how many variables we include in S, there is no test to confirm we have them all. Therefore, we have to be comfortable assuming we do based on expert knowledge for conditional exchangeability to hold. Note, in some cases, exclusion criteria can be set upfront to remove groups that don't have positivity (e.g. some groups always get treatment), but this should also be taken into account when making decisions based on the results.

If these conditions don't hold, alternative methods exist with different conditions. For example instrumental variable analysis, which attempts to use a predictor of treatment that behaves like a random assignment. More on this in the methods section later.

## Target trial

A good approach for ensuring a causal question is well defined is to first plan a hypothetical randomised experiment (target trial). For example, you can consider the causal effect of interest, eligibility criteria, the intervention(s), assignment mechanism, and outcomes. Presuming you can't actually run this hypothetical experiment, you can then consider how observational data can be used to emulate the target trial (e.g. do you have sufficient information to determine eligibility). If you're successful at emulating it, there's no difference.

By specifying the interventions, it's easier to spot those that are unreasonable (i.e. impossible to actually implement). For example, if we compare the risk of death in obese vs not-obese people at age 40, the implicit target trial is an instantaneous change in BMI. This is not realistic in the real world, meaning the counterfactual outcomes cannot be linked to the observed outcomes. We might not be able to specify the method used to lose weight, but a more realistic example could be a longitudinal study using BMI change over time as a treatment. If the interventions are not specified it's also harder to assess conditional exchangeability (may vary across interventions) and positivity (what combinations make treatment impossible).

If a target trial can't be specified, then associations from observational data are still interesting for generating hypotheses. However, it doesn't necessarily correspond to a causal effect.

---
## Causal Graphs

When addressing causal questions, visualising the relationships between variables can significantly aid understanding. A causal graph encodes assumptions about these relationships, making it easier to communicate and interpret them. It's important to also include unmeasured variables that may influence the model, so you can properly assess their potential impact. Typically, these graphs are drawn as directed acyclic graphs (DAGs), but even less formal versions can be very helpful. Each variable is represented as a "node", and arrows ("edges") between nodes indicate causal relationships, usually from left to right. Starting with just two variables, we would say they are independent if there's no arrow between them (e.g. the outcomes of rolling two random dice). However, once we start adding more variables, we have to consider the possible paths between them and whether they are open or blocked. We expect that any variable is independent of it's non descendants, conditional on it's parents. If all paths are blocked once we control for other variables, then the variables are conditionally independent. For example, symptom A might be independent of symptom B once we know the disease (it's not adding any additional information). However, keep in mind the graph shows a relationship if it applies to any individual from the population. Though not common, it's possible the average effect or association is not seen in a sample of the population (e.g. when positive and negative effects perfectly cancel out).

### Key Causal Pathways

There are three key types of paths to look for: mediators, confounders, and colliders. 

<img src="/assets/mediator_collider_confounder.png" alt="Mediator vs Confounder vs Collider" width="40%"/> 

#### 1. Mediator

A variable that lies on the causal pathway between two other variables. Conditioning on a mediator helps clarify the mechanism by which one variable affects another. Consider a situation where **Sales** are influenced by both **Ad Spend** and **Web Traffic**:

$$ \text{Ad Spend} \rightarrow \text{Traffic} \rightarrow \text{Sales} $$

Here, increasing **Ad Spend** increases **Traffic**, which in turn boosts **Sales**. If we condition on **Traffic**, the relationship between **Ad Spend** and **Sales** disappears because **Traffic** fully mediates this relationship. Mathematically:

$$ S \perp \perp A \mid T $$

If we fit a regression model without considering traffic, we might mistakenly conclude there’s a direct relationship between ad spend and sales. Once traffic is included, ad spend might no longer be statistically significant, as its effect on sales is mediated by traffic. We might therefore want to optimise ad spend to maximise traffic, using a separate model to predict the effect of the increased traffic on sales. However, if we wanted to estimate the total effect of ad spend on sales, we wouldn't want to control for traffic as this blocks the indirect path (underestimating the total).

#### 2. Confounder

A variable that causes two other variables. Conditioning on this variable is necessary to avoid a spurious association between the two other variables, biasing the causal effect estimate. Imagine that **Smoking** has a causal effect on both **Lung Cancer** and **Carrying a Lighter**:

$$ \text{Lighter} \leftarrow \text{Smoking} \rightarrow \text{Lung Cancer} $$

While carrying a lighter doesn’t cause lung cancer, if we compare lung cancer rates between people who carry a lighter and those who don’t, we might find a spurious association. This happens because smoking influences both lung cancer and carrying a lighter. To avoid bias, we must condition on smoking, breaking the spurious link between carrying a lighter and lung cancer ($$ Lighter \perp \perp Lung Cancer \mid Smoking $$).

In a sales example, if both ad spend and sales increase during the holidays due to seasonality (**Ad Spend ← Season → Sales**), failing to control for the season could lead to overstating the effect of ad spend on sales.

#### 3. Collider

A variable influenced by two or more other variables. Be cautious not to condition on a collider as it can create a spurious relationship between the influencing variables. In this example, both **Ad Spend** and **Product Quality** affect **Sales**, but they are independent of each other:

$$ \text{Ad Spend} \rightarrow \text{Sales} \leftarrow \text{Product Quality} $$

Once we condition on sales, we create a spurious association between ad spend and product quality ($$ Ad Spend \perp \perp Product Quality$$ but $$ Ad Spend \not\perp Product Quality \mid Sales $$). When sales are high, both ad spend and product quality may be high, and when sales are low, both may be low. This creates a false impression that ad spend directly improves product quality, even though both only influence sales. 

### General Rules

Conditioning on non-colliders (confounders or mediators) blocks the path, making the other variables independent. Conditioning on a collider (or its descendants) creates a dependency between the variables influencing it, which opens the path. Not conditioning on the collider blocks the path, keeping the influencing variables independent.

<img src="/assets/open_closed_paths.png" alt="Paths" width="90%"/>

By visualising causal relationships in a graph, you can better understand how variables are connected and whether observed associations are due to direct causal effects, confounding, mediation, or colliders. By distinguishing between these different relationships, we can handle them appropriately to have more accurate models and inference. While you don’t always need a graph, as the number of variables involved increases it can make it easier to manage. For example, let's look at just a slightly more complex graph with both confounders and colliders. By visualising the graph it's easier to identify whether X and Y are independent based on whether there's an open path between them:

<img src="/assets/open_closed_paths_complex.png" alt="Paths" width="100%"/>

Finally, let's look at a labelled example. Suppose we want to understand the direct causal relationship between the number of police officers and the number of victims of violent crimes in a city. Depending on what variables we control for, we open different paths and will get different results:

<img src="/assets/crime_graph_trio.png" alt="Crime graph" width="100%"/>

   1. We don't control for the mediator, which may overestimate the direct effect, or the confounder, which could cause an incorrect conclusion (more police causes more crime?). By not controlling for the collider, this path remains closed.
   2. We control for the mediator and the confounder, so will get a more accurate understanding of the causal effect. However, we open the collider path, so we may see a spurious association between media budget and the number of police.
   3. This is the more optimal set up. We control for the mediator and the confounder, but not the collider. Now if we include an irrelevant variable like media budget, it shouldn't cause an issue for estimating the effect of the number of police (though in reality we should exclude it given we don't think it causes our outcome). Also note, that if we wanted the total effect, we shouldn't control for the mediator.

### Challenges

In reality, the relationships between variables in causal models can be complex, making it challenging to identify the role of each variable. This is especially true when the time ordering of variables is unclear. Let’s consider the graph below looking at the causal effect of PFAS levels on health outcomes, controlling for obesity both before and after exposure:

<img src="/assets/pfas_health_graph.png" alt="PFAS effect" width="60%"/>
Source: [Inoue et al., 2020](https://doi.org/10.3390/toxics8040125)

In the graph, we see a bi-directional relationship between obesity and PFAS levels:
- **Pre-obesity** (obesity before PFAS exposure) can cause higher PFAS levels because PFAS accumulate in fat tissue. Additionally, obesity itself is associated with worse health outcomes. Therefore, **obesity\_pre** is a **confounder** because it affects both PFAS levels and health outcomes.
- PFAS exposure may also increase the risk of developing obesity (e.g. due to endocrine disruption or metabolic changes). In this case, **obesity\_post** (obesity after PFAS exposure) is a **mediator**, as it is part of the causal pathway from PFAS exposure to health outcomes.
- Obesity, PFAS levels, and health outcomes share common causes, such as dietary and lifestyle factors. These factors can affect both obesity and PFAS levels, and so they are also confounders.

To estimate the total effect of PFAS exposure on health outcomes, we want to control for confounders but avoid controlling for mediators, as doing so would block the indirect effect of PFAS on health outcomes via obesity. This is where the issue arises because both obesity and PFAS levels are cumulative measures (i.e. they reflect long-term, prior exposure). Most studies only measure obesity at baseline, and the temporality between PFAS exposure and obesity is not clear. The authors of this graph note that conditioning on obesity\_post (post-PFAS exposure) could: 1. Close the mediating path, thus underestimating the total effect of PFAS on health outcomes; and, 2. Introduce collider bias, opening a backdoor path between PFAS levels and health outcomes via the uncontrolled common causes of obesity\_post (shown by the dotted line in the graph), potentially leading to spurious associations. They therefore suggest not stratifying on obesity status if it's likely a mediator of the PFAS-health relationship being studied (as the stratum effects could be biased). On the other hand, it's generally safe to adjust for the common causes of PFAS and obesity (e.g. sociodemographic factors, lifestyle, and dietary habits) because these factors are less likely to act as mediators in this causal pathway. 

This is just one example, but it highlights the need to carefully consider the causal graph, and not over-adjust in ways that could bias the causal effects. When you want to measure the total effect, focus on controlling for confounders, not mediators.

## Identifying bias

When there's an association between the treatment and outcome that is not caused by the effect of the treatment on the outcome, there's a systematic bias. For example, when the treated and untreated groups are not exchangeable (bias under the null), there will be an association between the treatment and the outcome even if there's no causal effect. Bias under the null due to lack of exchangeability is caused by two structures: common causes (confounding); and, conditioning on common effects (selection bias). Causal graphs provide a useful way to search for these sources of bias, given lack of exchangeability translates to a lack of open paths between the treatment and outcome on the graph (other than those between A and Y). The third systematic bias type is measurement error (imperfectly measured variables), which can also cause bias under the null. 

### Confounding bias

In observational studies, the treatment is no longer randomised, and can be caused by multiple factors. This opens the risk that there are common causes of both the treatment and the outcome (confounders) which open a back door path. If you don't control for these, an association between the treatment and outcome can be due to a direct causal effect, but also the effect of those common causes. Some examples below of what this can look like:

<img src="/assets/confounding_bias_observational.png" alt="Confounding examples" width="100%"/>

To treat an observational study like a randomised experiment, we need conditional exchangeability to hold within each level of the measured covariates. If we know the causal DAG that produced the data (i.e. it includes all causes shared by the treatment and outcome, measured and unmeasured), then we can assess this using the backdoor criterion: "all backdoor paths between T and Y are blocked by conditioning on A, and A contains no variables that are descendants of T". In reality it's an uncheckable assumption. However, it's also important to consider the magnitude and direction of the bias. A large bias would require a strong association between the confounder and both the treatment and outcome. If the confounder is unmeasured, a sensitivity analysis can help quantify the potential impact (testing the impact of different magnitudes). It may also be possible to find surrogates which are associated with the unmeasured confounders (e.g. using income for socioeconomic status). These are not on the backdoor path (so can't completely eliminate the bias), but can indirectly adjust for some of the confounding. 

Other methods don't require conditional exchangeability, but have other unverifiable assumptions. For example, Difference-in-Differences (DiD) can be used with a negative outcome control (e.g. pre-treatment value of the outcome). Imagine the effect of Aspirin on blood pressure is confounded by unmeasured factors like history of heart disease. The pre-Aspirin blood pressure (C) is not causally related to the Aspirin (A), but may be associated due to the unmeasured confounders. Therefore, the difference between the average pre-treatment blood pressure for treated and untreated individuals can give us an idea of how much confounding is affecting the relationship between aspirin and blood pressure (Y):

$$E[Y_1 - Y_0 \mid A = 1] = \left( E[Y \mid A = 1] - E[Y \mid A = 0] \right) - \left( E[C \mid A = 1] - E[C \mid A = 0] \right)$$

This assumes that the confounding effect on the outcome Y is the same as the confounding effect on the negative outcome control C. Therefore, the choice of adjustment method just depends on which assumptions are more likely to hold for the study in question. Uncertainty is unavoidable with observational data, and often the best you can do is be explicit about your assumptions and use sensitivity tests to try quantify the potential impact of bias. This allows others to challenge them in a more productive way.  

### Selection bias

Selection bias refers to bias that arises when individuals or units are selected in a way that affects the generalisability of the results. This bias can occur when the process of selecting subjects leads to a sample that is not representative of the population of interest. Here we focus on selection bias under the null, meaning there will be an association even if there's no causal effect between the treatment and the outcome. This can occur when you condition on a common effect of the treatment and outcome, or causes of each. For example, when observations are removed due to loss to follow up, missing data, or individuals opting in/out. 

<img src="/assets/selection_bias_observational.png" alt="Confounding examples" width="100%"/>

Selection bias can also occur in the analysis of randomised experiments. For example, Conditional-on-Positives (CoPE) bias occurs when you only consider the "positive" cases for a continuous outcome, distorting the comparison between treatment and control. For example, if you split a continuous outcome like "amount spent" into a participation model (0 vs >0 spent) and a conditional amount spent model (for those >0) to handle over-saturation at 0. The conditional amount spent model ignores the fact that treatment can move some people from zero spending to positive spending, meaning the "zero spenders" and the "spenders" in the treatment group are systematically different to those in the control. By comparing the "spenders" in treatment and control, we get the causal effect plus the selection bias (i.e. $$E[Y_0 \mid Y_1 > 0) - E[Y_0 \mid Y_0 > 0]$$). Most likely, the "spenders" in the control are inflated compared to the treatment group (negative bias), as those who only spent because of the treatment likely spend less than those who spend regardless (different baseline). We need to account for the fact that treatment affects both the participation decision (whether someone spends anything) and the spending amount (conditional on spending).

Other than design improvements, selection bias can sometimes be corrected with IPW. As with confounding, we also have to consider the expected direction and magnitude of the bias to understand the impact.

## Random variability

Putting aside now the sources of systematic bias (confounding, selection, measurement) which are issues of identification, the remaining challenge is random variability due to sampling (i.e. estimation). We can think of an estimator as a rule that takes sample data and produces a value for the parameter of interest in the population (the estimand). For example, we calculate the point estimate $$\hat{P}(Y=1 \mid A=a)$$ for the population value $$P(Y=1 \mid A=a)$$. Estimates will vary around the population value, but an estimator is considered consistent if the estimates get closer to the estimand as the sample size increases. We therefore have greater confidence in an estimate from a larger sample, which is reflected in a smaller confidence interval around it. Standard methods are also used to calculate confidence intervals for standardised and IP weighted effects, just accounting for the weighting. However, these only quantify the uncertainty due to random variation, so it doesn't account for any systematic biases (the uncertainty of which doesn't decrease as the sample size increases). 

This also true in experiments. Random assignment of the treatment doesn't gaurantee exchangeability in the sample, but rather it means any non-exchangeability is due to random variation and not systematic bias. Sometimes you're very unlucky and approximate exchangeability doesn't hold (e.g. you find a significant difference in one or more pre-treatment variables that are confounders). There's ongoing debate about whether to adjust for chance imbalances in randomised experiments, particularly for pre-treatment variables that were not part of the randomisation procedure. The basic conditionality principle tells us we can estimate ATE without adjusting for covariates because treatment is randomised (unconfounded), but the extended version justifies conditioning on any pre-treatment X that are not randomised to improve precision (by explaining outcome variance), and estimate sub-group effects (heterogeneity). In practice, a pragmatic approach to avoid aver-adjustment is to adjust only for strong predictors of the outcome that are imbalanced enough to plausibly affect inference.

## Target population

--- TODO


---
# Models

As we covered earlier, estimates will vary around the population value, but an estimator is considered consistent if the estimates get closer to the estimand as the sample size increases. Therefore, as you add more treatment levels, the estimate is still unbiased, but the probability the sample average is close to the population average decreases (confidence interval widens). If T is continuous (e.g. a dosage), then we run into the problem of not having observations for all values of T, and we need to supplement the data with a model (estimator). This is not specific to causal inference, but rather about estimating a population parameter from a sample. 

A conditional mean model specifies a functional form for $$E[Y \mid X=x]$$, meaning it models the expected value of the outcome Y given covariates X. Models make predefined assumptions about the joint distribution of the data. Parametric conditional mean models use these assumptions to compensate for the limited information in the observed data (e.g. when some levels of treatment don't exist in the sample). Different models have different functional forms, which restrict the shape of the conditional mean function. For example, linear models (e.g. $$E(Y \mid A) = \theta_0 + \theta_1 A$$) restrict the conditional mean function to a straight line, allowing the slope and intercept to vary based on their parameters ($$\theta_0$$ and $$\theta_1$$). By adding more parameters (e.g. higher order terms), the straight line can be curved, and becomes more wiggly as the number of parameters increases. For example, if there are as many parameters as observations, then there also the same number of inflection points (i.e. over-fitting the data). The larger the number of parameters, the less smooth the model will be as it's less restrictive, and the less likely it is to be biased. Inversely, the fewer the parameters, the smoother it will be. However, there's a bias-variance trade-off: increasing the parameters reduces the bias but increases the variance (widening the confidence intervals). 

Common conditional mean models:
- Generalised Linear Model (GLM) which have 2 components: a linear functional form; and a link function which constrains the output values (e.g. log for y>0; logit for y between 0 and 1; identity for continuous y);
- Generalized Estimating Equations (GEE): a method for estimating population-averaged (marginal) effects in correlated or clustered data (e.g. multiple observations over time or within a cluster);
- Genarlised additive model (GAM) is an extension of the generalized linear model (GLM) that relaxes the assumption of linearity in the covariates. Rather than assuming a linear relationship between predictors and the response, GAMs allow for smooth, potentially nonlinear effects of each covariate.
- Kernel regression: Kernel regression is a nonparametric technique that estimates the conditional expectation $$E[Y \mid X=x]$$ by averaging nearby observations of Y, with weights decreasing with distance from the target point x. It does not assume that the relationship between X and Y is linear, quadratic, or of any specific form — making it fully data-driven.

Though not specific to causal inference, parametric models are only correct if there is no model misspecification. However, some misspecification is inevitable, so careful consideration is required. For example, testing assumptions with different model specifications (sensitivity analysis). The assumptions of the model are in addition to the identifiability assumptions discussed earlier. Conditional mean models are not inherently causal but become causal estimators under identifying assumptions. This extends our assumptions required for valid causal inference to the following:
- Exchangeability
- Positivity
- Consistency
- No measurement error
- No model misspecification




Leveraging the above, the following are some of the commonly used causal inference methods:

- Matching, Propensity Scores, Outcome Regression → All focus on balancing observed confounders.
- IV, G-estimation (SNMMs) → Handle unobserved confounding (IV for point exposure, G-estimation for time-varying exposure). Mathematically similar when IV is used for a point treatment; G-estimation generalizes to longitudinal, dynamic treatments.
- G-methods family → Includes IPW, G-computation, and G-estimation. These are interrelated strategies designed for longitudinal/time-varying treatments.
- DiD / TWFE / Synthetic Control → Focused on panel or repeated cross-sectional data, addressing time trends and group-level unobserved confounding.
- RDD → Solves selection on observables at a cutoff, can be combined with DiD (RDDiD) for discontinuities in time trends.

- Outcome regression
- Matching
- Propensity scores (incl. IPW)
- IV
- DiD
- TWFE
- Synthetic control
- RDD


Not done yet:

- Standardisation
- G-estimation
- IPW
- Causal survival analysis
- Meta learners
- De-biased orthogonal ML (Double Machine Learning)
- Synthetic DiD

Geo-experiments: where does it fit?

Comparison

Evaluating models




### Example

Imagine you work on the Apple Watch’s breathing app and want to understand whether it lowers users’ resting heart rate (RHR) over a two-week period. Some users engage with the app frequently, while others don’t use it at all—but these groups likely differ in important ways. For instance, breathing app users may be more health-conscious, younger, and exercise more, but also more stressed (which might drive them to use the app). These factors can also affect RHR, making them potential confounders. 

To make a fair comparison, we use matching to balance these observed characteristics across frequent and infrequent users. We match on pre-treatment variables that proxy for confounders: baseline RHR, heart rate variability (as a stress indicator), hours of sleep, daily steps, and weekly workouts (as measures of fitness). We also control for the time of day RHR is measured to reduce noise in the outcome. However, we avoid adjusting for post-treatment variables like sleep, fitness, or app engagement, since these could be colliders—affected by both app usage and RHR. We also avoid controlling for mediators like post-treatment stress, which lie on the causal path from treatment to outcome. 


---
# Estimating causal effects with models

## Regression

Under strong assumptions (unconfoundedness), regression models can be used to estimate causal effects such as ATE. For example, using a binary variable to indicate whether an observation was treated or not (T), we get:

$$Y_i = \beta_0 + \beta_1 T_i + \epsilon_i$$

Where:
- $$\beta_0$$ is the average Y when $$T_i=0$$ (i.e $$E[Y \mid T=0]$$)
- $$\beta_0 + \beta_1$$ is the average Y when $$T_i=1$$ (i.e. $$E[Y \mid T=1]$$) 
- $$\beta_1$$ is the difference in means (i.e. $$E[Y \mid T=1] - E[Y \mid T=0]$$)
- $$\epsilon$$ is the error (everything else that influences Y)

In this single variable case, OLS is computing the difference in group means, scaled by how much variation there is in treatment assignment. If T is randomly assigned then $$\beta_1$$ unbiasedly estimates the ATE:

$$ATE = \beta_1 = frac{Cov(Y_i, T_i)}{Var(T_i)}$$

However, when using observational data, the key assumption of exchangeability often fails ($$E[Y_0 \mid T=0] \neq E[Y_0 \mid T=1]$$). To address this, we can extend the model to control for all confounders $$X_k$$ (i.e. variables that affect both treatment and outcome):

$$Y_i = \beta_0 + \beta_1 T_i + \beta_k X_{ki} + \epsilon_i$$

Now $$\beta_1$$ estimates the treatment effect conditional on the confounders (conditional difference in means), meaning it adjusts for the differences between the treated and untreated observations. To understand how it works, think of it in 2 steps: 1) regress T on X to isolate the variation in T not explained by the confounders (i.e. the residuals, which are orthogonal to X by design); 2) regress Y on the residuals from step 1 (the part of T that isn't explained by X). If T was randomised, then X would have no predictive power and the residuals would just be the original T. Note, the normal equation does this in one step for multiple regression using the inverse matrix ($$\left(X^\top X\right)^{-1}$$) to "adjust" each predictor’s covariance with Y for the correlations among the predictors. 

### How it works (intuitive example)

If conditional exchangeability holds ($$Y_0, Y_1 \perp\!\!\!\perp T \mid X$$), regression can estimate the ATE by controlling for X. When X is binary, this works by estimating the treatment effect within each stratum and averaging them to get a single ATE for the whole dataset. But the weights are not based on sample size, rather they depend on treatment variation within each group. For example, suppose X indicates whether a user is on mobile or desktop. There are 800 mobile users (790 treated, 10 control) and 200 desktop users (100 treated, 100 control). The treatment effect on mobile is 1 and on desktop is 2. Even though mobile users make up 80% of the sample, their group has almost no treatment variation, so regression gives their effect little weight. Instead, the balanced desktop group carries more influence. The final ATE will be something like 0.3×1+0.7×2=1.7, showing that strata with more treatment variation contribute more to the overall estimate.

### Selecting controls

To have an unbiased estimate we need to include all confounders in the model. Adding predictors of the outcome can also be useful for improving power (reduces variance in the outcome Y, making it easier to find the treatment effect). However, we want to be careful not to over-adjust by adding many variables to the model as some variables can induce bias, or reduce the power:

❌ Pure predictors of the treatment: including these can reduce the variability of treatment, inflating the standard error (but not biasing the estimate if they're not related to the outcome).

❌ Mediators: variables on the causal path between treatment and outcome. Controlling for these blocks part of the treatment effect, underestimating the total effect.

❌ Colliders: variables that are caused by both treatment and outcome (or their causes). Controlling for colliders opens a backdoor path, introducing spurious associations. For example, controlling for job type when estimating the effect of education on income may bias the estimate, because job type is influenced by both education and income.

### Example

## TODO Add example code

### Causal strength

While regression can estimate causal effects, it does not prove causality on its own. Its reliability hinges on the validity of the assumptions:

- **Exchangeability** (no unmeasured confounding): conditional on covariates X, the treatment assignment is as good as random. Violations result in omitted variable bias. This is not a testable assumption as we can never be sure all confounders are measured and correctly included.
- **Positivity** (overlap): Positivity requires that for every combination of covariates X, there is a non-zero probability of receiving both treatment and control. This ensures that we can make meaningful comparisons between treated and untreated units within every subgroup of X. Violations result in extrapolation as regression models predict effects for units outside the support of the data, which can be unreliable. Pre-processing to restrict the population of interest might be possible.
- **Consistency**: The treatment is well-defined and uniformly applied. Violations lead to ambiguity about what effect is being estimated, weakening causal interpretation. This can be more challenging with observational data, with violations potentially being more hidden when regression is used (e.g. controlling for post-treatment variables could remove part of the treatment effect).

## Matching

Matching tries to approximate a randomised experiment by directly comparing similar observations. Meaning for each treated observation we're looking for a similar (exchangeable) untreated observation to compare to based on some set of covariates X. Exact matching (finding someone with exactly the same covariates) is often hard, especially as the size of X increases. Therefore, approximate matching methods are more common, for example matching with the "nearest neighbour" based on a defined measure of distance (e.g. euclidean or mahalanobis). Other common matching methods include propensity score matching (PSM), Caliper matching, and Genetic or optimal matching. Once we have our matched data, we can then compare the outcomes of the treated and untreated observations to get the ATE (with each matched pair contributing equally to the estimate). However, there are two key sources of bias to consider: 1) it only removes bias from observed confounders (like regression); and, 2) unless matched observations are identical on all covariates (exact match), the differences introduce bias (slowly diminishes as the sample size increases). We need to include and measure correctly all confounders for the first bias, but the second can be corrected by fitting a model (e.g. regression) on the matched data to adjust for the small differences.

### Comparison to regression

Matching adjusts for confounding by controlling for observed covariates like in regresion. However, regression models the outcome and adjusts for covariates in estimation, whereas matching first selects a balanced subset of the data based on the covariates and then does a direct comparison.

### Curse of dimensionality and bias adjustment

The more variables we use, the larger and more sparse the feature space. As a result, matched observations move further apart the more variables we add. This introduces bias in our estimates because the similarity of the matched points decreases. As the sample size grows, the bias decreases, but this happens slowly. Linear regression gets around this by first projecting points onto the outcome and then making the comparison on the projection. 

A solution to this problem is to perform the matching, and then fit a mode like linear regression on the matched data to adjust for the differences. This is different to just fitting a regression model:
- Matching is non-parametric — it makes fewer assumptions than linear regression (like linearity or additivity). This can make matching more robust, especially in the presence of strong non-linearities.
- The regression model is only used locally — to adjust for small differences between a treated unit and its match. It’s not used to extrapolate across treatment groups.

### Distance

Note, for some measures it's important to standardise features first so they're on a similar scale. However, 

### Methods

Exact matching: Match only units with the same values for certain key variables.
Nearest neighbor matching: Find the control unit closest to each treated unit.
Propensity score matching: Match based on the probability of receiving treatment.

Here's how a typical matching workflow might look:

Select covariates that capture potential confounders.

Choose a matching method (e.g., nearest neighbor with or without replacement).

Check covariate balance before and after matching. Standardized mean differences can help here.

Estimate the treatment effect on the matched sample (e.g., difference in means).

### Weighting

Both regression and matching aim to adjust for confounding by controlling for covariates. But they differ in how they weight data. Each matched pair contributes equally to the estimate, regardless of how common or rare treatment is in that group. However, treatment variation still indirectly affects the analysis: in groups where most units are treated and few controls are available, it becomes difficult to find good matches. Unmatched or poorly matched treated units contribute little or nothing to the final estimate.

For example, take our earlier example with 800 mobile users (790 treated, 10 control) and 200 desktop users (100 treated, 100 control), where the treatment effect is 1 for mobile and 2 for desktop. Regression gives more weight to the desktop group because it has more treatment variation. Matching also gives less weight to the mobile group, but for a different reason: with only 10 controls, few of the 790 treated mobile users can be well matched. In contrast, the balanced desktop group yields many good matches that heavily influence the estimate. In short, regression emphasizes treatment variation, while matching emphasizes comparability and match quality.

### Limitations

Matching only adjusts for observed covariates. Hidden confounders can still bias results.

Poor overlap between groups means fewer good matches.

Matching discards data, which can increase variance.

Quality of results depends on careful covariate selection.


## Propensity scores

Instead of controlling for all confounders $$X$$, which can be high-dimensional, another option is to create a single balancing score as a function of $$X$$, $$b(X)$$, and control for that instead. The most common example is the propensity score:

$$
e(X) = \mathbb{P}(T = 1 \mid X)
$$

This is the probability of receiving treatment given your observed characteristics $$X$$. If treatment assignment was not random, we expect the probability of treatment will vary depending on X. However, assuming X contains all confounders, $$T$$ is as good as random once we control for $$X$$:

$$
(Y(0), Y(1)) \perp T \mid X
$$

If that’s true, then:

$$
(Y(0), Y(1)) \perp T \mid e(X)
$$

In other words, the propensity score captures all the information in $$X$$ that makes treatment assignment unconfounded. So instead of adjusting for all of $$X$$, we can just adjust for $$e(X)$$. Now, if two people have the same propensity score, then whether one got treated and the other didn’t is essentially random. So rather than trying to find people who are similar across all covariates, we just need to find those with similar scores. This is how propensity scores make matching, stratification, or weighting a lot more manageable.

### Conditional mean difference

Recall the conditional mean difference (difference in outcomes between the treated and untreated):

$$\mathbb{E}[Y \mid X, T = 1] - \mathbb{E}[Y \mid X, T = 0]$$

Re-writing this using the propensity scores, the left hand side becomes:

$$\mathbb{E}\left[ \left. \frac{Y}{e(x)} \,\right|\, X, T = 1 \right] P(T)$$

We take the outcomes of treated people, reweight them using the inverse of their treatment probability ($$\frac{Y}{e(x)}$$). This downweights those with a high chance of treatment and upweights those with a low chance so we don't over-represent those who were likely to be treated. This is then scaled by how common treatment is overall to give us an estimate of what everyone would have experienced, had they all received the treatment.

Then the right hand side follows the same idea for the untreated:

$$\mathbb{E}\left[ \left. \frac{Y}{1 - e(x)} \,\right|\, X, T = 0 \right] (1 - P(T))$$

This uses $$1-e(x)$$ in the denominator (the probability of not being treated, given X), so $$\frac{Y}{1-e(x)}$$ gives more weight to those unlikely to be untreated (but were). Scaling by how common it is to be untreated (1-P(T)), this estimates what the entire population would have experienced if no one had received the treatment, by appropriately reweighting the observed outcomes of untreated individuals.

Putting these together and simplifying, the expected potential outcomes under treatment and control are estimated as:

$$
\mathbb{E}\left[ \frac{T \cdot Y}{e(X)} \right] \quad \text{and} \quad \mathbb{E}\left[ \frac{(1 - T) \cdot Y}{1 - e(X)} \right]
$$

Subtracting these gives an estimate of the Average Treatment Effect (ATE):

$$
\mathbb{E}\left[ \frac{T \cdot Y}{e(X)} - \frac{(1 - T) \cdot Y}{1 - e(X)} \right]
$$

This can also be written more compactly as:

$$
\mathbb{E} \left[ Y \cdot \frac{T - e(X)}{e(X)(1 - e(X))} \right]
$$

This technique is called Inverse Probability Weighting (IPW). It adjusts for confounding by giving more influence to people who are "underrepresented" in their group: those treated despite low probability, or untreated despite high probability. We're creating two comparable synthetic populations (each the same size as the original), one where everyone is treated and one where no one is. Comparing them gives us the conditional mean difference using only $$e(X)$$ instead of the full set of covariates. Given the propensity score is an estimation, we need to account for this in the standard error of the ATE (e.g. by bootstrapping an ATE sampling distribution).

### Estimating e(X)

With observational data, we typically don't know the treatment assignment mechanism, so we predict the propensity score using a model like logistic regression. We’re predicting treatment from the confounders, but the goal isn’t to make the best prediction, it’s to balance the sample and remove confounding. That means better predictive accuracy doesn’t necessarily help. For example, if you include variables that are predictive of treatment but unrelated to the outcome, your model’s predictions may improve, but the variance of your ATE estimate will increase. The same thing happens in regression if you add irrelevant variables.

### Positivity

It’s important that treated and untreated units overlap. That is, everyone must have some chance of receiving and not receiving treatment: $$0 < e(X) < 1$$. This is known as the positivity assumption. You can check this by comparing the distributions of propensity scores in the treated and untreated groups. Ideally, they should substantially overlap across the full range, and be smooth, with no scores close to 0 or 1. Propensity scores near 0 or 1 mean someone is almost guaranteed to be treated (or not), which causes the inverse probability weight to blow up, increasing the variance of the ATE (makes it sensitive to a few influential observations). As a rule of thumb, no individual should get a weight over 20. Some will trim the propensity scores to limit the range (e.g. 0.05 to 0.95), but this can introduce bias. Another option is to trim observations outside the overlap region (i.e. drop treated with no comparable untreated match), but this limits the generalisability.

<img src="/assets/propensity_overlap.png" alt="Overlap example" width="95%"/>

### Propensity scores + regression (doubly robust)

Propensity scores can be combined with an outcome model (e.g., regression) to produce a doubly robust estimator. The key advantage is that you get a consistent estimate of the average treatment effect (ATE) if either the outcome model or the propensity score model is correctly specified. Since both methods alone have weaknesses (outcome models can be biased, and propensity score weighting can be unstable), combining them lets one "lean on" the other when it is misspecified.

For example, consider the Augmented Inverse Probability Weighting (AIPW) estimator:

<img src="/assets/aipw_formula.png" alt="AIPW formula" width="60%"/>

Where:
- $$T_i$$ is the treatment assignment (1=treated)
- $$\hat{\mu}_1(X) = \mathbb{E}[Y \mid X, T = 1]$$ is the predicted outcome from the outcome model for the treated (trained on T=1 only)
- $$\hat{\mu}_0(X) = \mathbb{E}[Y \mid X, T = 0]$$ is the predicted outcome from the outcome model for the untreated (trained on T=0 only)
- $$\hat{P}(X) = \mathbb{P}(T = 1 \mid X)$$ is the propensity score model (trained on all)

We create two synthetic samples to estimate potential outcomes: one where everyone is treated (estimating Y(1)), and one where everyone is untreated (estimating Y(0)). The outcome model provides baseline predictions ($$\hat{\mu}_1(X), \hat{\mu}_0(X)$$), then the residuals (actual minus predicted outcomes) are weighted by the inverse propensity score for the units actually observed in that treatment group. This weighting gives more influence to residuals from units with unlikely treatment assignments, helping correct bias especially in regions with sparse data. If a unit was not treated, its weighted residual is zero (due to the $$T_i$$ in the numerator), so the estimate for that unit relies on the outcome model prediction alone. Because the variance of the AIPW estimator reflects uncertainty from both models plus residual noise, bootstrapping is the simplest and most robust way to estimate standard errors.

#### Example

Imagine we want to estimate the effect of a promotion on spending for all users, but income (an important confounder), is imperfectly measured. High-income customers in our data were more likely to get the promotion and tend to spend more, so a naive regression may overestimate the promotion effect. If some customer types are rarely treated, outcome predictions there may be poor. By weighting the residuals inversely by the probability of receiving treatment, we "rescale" these residuals to reflect the overall population better, providing a correction in underrepresented regions. The residual tells us how wrong the outcome model was, and the weighting ensures this correction is applied in proportion to where the model is likely to be wrong (i.e. in underrepresented areas).

- **If the outcome model is good**: the residuals are small, so the correction term adds little, and the estimate relies on the correct outcome model predictions (i.e. it doesn't matter if the propensity scores are wrong).
- **If the outcome model is poor**: the residuals are large, and the correction (weighted by the correctly specified propensity score) helps adjust bias, making the estimator rely more on propensity weighting. Assuming the propensity model is correct (i.e. $$\mathbb{E}\big[ T_i - \hat{P}(X_i) \big] = 0$$), it reduces down to an IPW estimator ($$\frac{T_i Y_i}{\hat{P}(X_i)}$$).

This means we have two chances to get unbiased estimates: either the outcome model is correct, or the propensity score model is correct.

| Method                    | Assumptions                          | Robustness               | Efficiency                  | Overfitting Risk     | Notes |
|---------------------------|--------------------------------------|---------------------------|------------------------------|----------------------|-------|
| **Outcome Regression**    | Outcome model is correctly specified | ❌ Not doubly robust       | ✅ Typically low variance     | 🔶 Moderate (esp. with complex models) | Simple to implement, but biased if model is misspecified or if you control for post-treatment variables |
| **IPW**                   | Propensity model is correctly specified | ❌ Not doubly robust     | ❌ High variance (esp. with extreme scores) | 🔶 Moderate (due to unstable weights) | Can become unstable when propensity scores are near 0 or 1; trimming or stabilization often needed |
| **AIPW** (Doubly Robust)  | Either outcome or propensity model is correct | ✅ Doubly robust       | ✅ Efficient if both models are correct | 🔺 High unless using cross-fitting | Balances regression and reweighting; residual correction term protects against single-model bias; cross-fitting recommended to avoid overfitting |

### Propensity scores + matching

Propensity scores offer a solution to the curse of dimensionality in matching. Instead of matching on all covariates directly, we match on a single summary score: the propensity score, which estimates the probability of treatment given covariates. This reduces the dimensionality of the problem and ensures that units are matched based on their likelihood of receiving treatment. It focuses the comparison on units that are similar in terms of confounding risk.

A caliper sets a tolerance threshold for the maximum allowable difference in propensity scores for a pair to be matched. A common choice is 0.2 standard deviations of the logit of the propensity score, which strikes a balance between bias and variance.

However, there are some important caveats:
- **Overlap is still essential**: If treated and control units do not share common support (for example, if their propensity scores are close to 0 or 1), good matches may not exist. This can lead to biased estimates.
- **Bias-variance trade-off**: Tight matching using small calipers can reduce bias by ensuring closer matches, but it also increases variance because fewer units are retained. Looser calipers allow more matches and reduce variance, but at the risk of introducing bias through poor matches.
- **Standard errors require care**: The propensity score is estimated, not observed. Standard error calculations that ignore this will understate the true uncertainty. Adjustments or bootstrapping methods are typically needed.

Matching is intuitive and transparent. However, unlike doubly robust methods, it depends entirely on the quality of the propensity score model. If the model is misspecified, the resulting matches can still be unbalanced, and bias will remain.

### Causal strength

In summary, propensity scores collapse high-dimensional covariate adjustment into a single number. When assumptions hold, they let us remove bias from confounding. IPW in particular gives us a way to simulate what would have happened if everyone received or didn’t receive treatment.

It's reliability hinges on the validity of the assumptions:

- **Exchangeability** (no unmeasured confounding): Conditional on the observed covariates X, the treatment assignment is as good as random. This means that all confounders affecting both treatment and outcome are measured and properly accounted for. Violations of this assumption lead to omitted variable bias, as unmeasured confounders distort the estimated treatment effect. Because we can never be sure that all relevant confounders are captured, this is fundamentally an untestable assumption.
- **Positivity** (overlap): This means every unit has some chance of receiving both treatment and control — that is, the propensity score is strictly between 0 and 1. It ensures we can compare treated and untreated units within the same region of covariate space. When positivity is violated (e.g. some units are almost always treated or almost never treated), we end up extrapolating: models estimate effects in regions with no real data, which makes the results unreliable. In practice, it's common to inspect the overlap in propensity score distributions and possibly trim or restrict the sample to regions of good overlap to avoid extreme weights and unstable estimates.
- **Consistency**: As before, the treatment is well-defined and uniformly applied. 






## Inverse Probability Weighting (IPW)

This is a weighting method.

Goal: Create a pseudo-population in which treatment assignment is independent of covariates.

How: Weight each unit by the inverse of the probability of receiving the treatment they actually received.

Used For: Estimating ATE under unconfoundedness.


Stabilized Weights
A variation of IPW where weights are scaled to reduce variance:

Stabilized Weight= 
P(T=t∣X)
P(T=t)

Helps prevent extremely large weights from dominating the estimate.



## Standardization and the parametric g-formula

Alternative to IPW






## Doubly robust




## Instrumental Variables (IV) analysis

The methods described above all rely on the untestable assumption that all required adjustment variables have been included and correctly measured. Inevitably, this is wrong to some extent, resulting in residual bias. 

When unobserved confounders bias our estimates, we can't directly measure the causal effect of a treatment T on an outcome Y. Instrumental variable (IV) analysis offers a way to estimate this effect indirectly, under specific conditions. The key is to find an instrument (Z) that influences the treatment T, but has no direct or indirect effect on the outcome Y except through T. In other words, any change in Y that results from changes in Z must operate entirely through T. If this holds, we can isolate the causal effect of T on Y by dividing the effect of Z on Y by the effect of Z on T. However, if Z affects Y through other pathways (violating the exclusion restriction), then the observed changes in Y cannot be attributed solely to T, and the IV estimate is invalid.

<img src="/assets/iv_graph.png" alt="IV graph" width="60%"/>

### Key assumptions:

- **Relevance**: The instrument Z has a non-zero causal effect on the treatment T (Z->T), meaning $$Cov(Z,T) \neq 0$$. Intuitively, Z needs to move T enough for us to see the change in Y. This assumption is testable.
- **Exclusion Restriction**: Z affects Y only through T, meaning there's no direct or indirect effect of Z on Y, so we can assume any change in Y is due to T. This is not testable, so it must be justified theoretically.
- **Independence (exchangeability)** (also called Ignorability / Instrument Exogeneity): The instrument is as good as randomly assigned with respect to Y, meaning there's no open backdoor paths that could cause a spurious correlation. Given there are unmeasured confounders, this is not testable with data, so it also must be justified theoretically.
- **Monotonicity**: The effect of T on Y must be homogenous (only in one direction) and linear (in the case of a binary treatment this is guaranteed) to interpret it as a Local Average Treatment Effect (LATE). Meaning the estimates are specific to those who were shifted by the instrument (from Y=0 to Y=1, known as compliers). The never-takers and always-takers are unchanged ($$T_\text{i0}=T_\text{i1}$$), so IV says nothing about the effect for them. Though uncommon, we assume there are no defiers (those who do the opposite).

<img src="/assets/pop_groups_iv.png" alt="IV populations" width="60%"/>

Monotonicity is considered optional because it’s not essential for the identification of the causal effect; rather, it affects the interpretation of the estimate. If monotonicity holds, you get a clean interpretation of the treatment effect as the LATE for the compliers. If the compliers are not special in any way, the results should generalise well to the whole population, but it's not gauranteed like in a randomised experiment. If monotonicity doesn’t hold, the IV estimate becomes more complex, representing a weighted average of treatment effects (WATE) across different groups (rather than a uniform effect), making it harder to interpret and generalise.

### How it works

As described above, the association of the instrument Z with the outcome Y will be a linear multiple of the association of the instrument Z with the treatment T:

<img src="/assets/iv_graph_formulas.png" alt="IV formulas on graph" width="40%"/>

- **First stage effect (α)**: Regress T on Z to estimate how much the instrument shifts the treatment (the part of T explained by Z). This gives the first-stage coefficient α, where a 1-unit increase in Z causes an α-unit change in T.
- **Reduced-form effect ($$\Delta Y = α*β$$)**: Regress Y on Z to estimate the total effect of the instrument on the outcome. Since Z moves T by α, and T causes a change in Y by β, the overall change in Y due to Z is α⋅β. This is called the reduced form because it skips over the treatment step.
- **Solving for β**: Re-arranging the above formula gives us the effect of T on Y (β): $$\beta = \frac{\Delta Y}{\alpha} = \frac{\text{Reduced form}}{\text{First stage}}$$.

Since IV estimates rely on marginal differences (i.e. changes in T and Y), we can (and should) use linear regression even if the outcome is binary or categorical. The goal is not to perfectly model Y, but to isolate variation in T that is exogenous (i.e. uncorrelated with confounders), using Z. Importantly, linear regression in the first stage (T ~ Z) ensures that we isolate the part of T that is orthogonal to unobserved confounders (the error). 

#### Wald Estimator 

For a single binary instrument and binary treatment, the Wald estimator directly applies the logic above:

$$\hat{\beta}_{\text{IV}} = \frac{E[Y \mid Z = 1] - E[Y \mid Z = 0]}{E[T \mid Z = 1] - E[T \mid Z = 0]} = \frac{\alpha \cdot \beta}{\alpha} = \frac{\text{Reduced form}}{\text{First stage}}$$

If the instrument perfectly predicts treatment (i.e. 100% compliance), the denominator is 1, and the estimator reduces to the difference in outcomes (like a standard randomized controlled trial). However, if the instrument is weak (i.e. α is close to 0), the denominator becomes small, leading to unstable and biased estimates.

#### Two-stage least squares (2SLS)

In practice, 2SLS is the standard method for IV estimation (e.g. IV2SLS in Python’s linearmodels):

- **First stage ($$\hat{T}$$)**: Same as before, regress T on Z. However, now we take the predicted treatment values as the output ($$\hat{T}$$);
- **Second stage ($$\beta$$)**: Regress Y on $$\hat{T}$$ to isolate the component of T that is attributable to Z (i.e. removing variation), removing variation due to confounders.

In the simple case of one instrument and no covariates, 2SLS produces the same result as the Wald estimator. But unlike Wald, 2SLS extends to: multiple instruments; continuous treatments and instruments; the inclusion of control variables (added to both stages); and valid standard errors (accounts for the fact $$\hat{T}$$ is an estimate and not an observed value). 

#### Example

Let's solidify this logic with a simplified example. Suppose we want to estimate the effect of ad exposure (T) on purchases (Y). A temporary bug (Z) in the ad system randomly caused some users to be 30 percentage points less likely to see ads (first stage effect: α=0.3). Those affected by the bug (Z=1) had a 1 percentage point lower conversion rate during the session (reduced form effect: α⋅β=0.01). This tells us that a 30 percentage point decrease in ad exposure leads to a 1 percentage point decrease in purchases. Using the Wald estimator to scale up, the effect of 100% of users not seeing ads is estimated as $$\frac{0.01}{0.3}=0.03$$, meaning removing all ads would cause a ~3 percentage point drop in conversion. 

<img src="/assets/iv_graph_formulas_example.png" alt="IV formulas on graph" width="40%"/>

This is an overly simplified example: we assume all IV assumptions hold. In reality, we’d need to test exchangeability (e.g., did the bug disproportionately affect certain segments or device types? Are the periods comparable and generalizable? Could users have noticed the change and reacted differently?). If there are measurable confounders of Z and Y, we can add these as covariates in 2SLS (e.g. device type).

### Selecting an instrument

Finding a valid instrument is challenging in practice. In industry, common sources of instruments include bugs, experiments (especially unrelated A/B tests that nudge compliance with treatment), feature rollouts, and external shocks such as policy changes or outages (i.e. natural experiments). While these are commonly available, recall they need to have a strong enough effect on the treatment to be viable (i.e. the first stage estimate must be significant). It's also possible to use user, device, or product traits as instruments (e.g. OS type, app version), but these are often directly or indirectly correlated with outcomes of interest (e.g. conversion rates), violating the exclusion restriction.

 Some examples of instruments for inspiration:

- In this [study](https://wol.iza.org/uploads/articles/250/pdfs/using-instrumental-variables-to-establish-causality.pdf) a policy reform that raised the minimum school-leaving age was used as an instrument to estimate the effect of education on wages. Direct comparisons of students by years of schooling would be confounded by factors like ability. But assuming cohorts before and after the reform are comparable, and wages weren’t otherwise affected by the policy, the change in average earnings reflects the causal effect of an additional year of education.

- At Uber ([hypothetically](https://eng.uber.com/causal-inference-at-uber/)) a bug was used as an instrument to study the effect of delivery delays on customer engagement. Comparing delayed vs. non-delayed orders is biased by factors like order volume. But if the bug randomly introduced delays and had no direct effect on engagement, it isolates the causal effect of delay.

- [Roblox](https://corp.roblox.com/it/notizie/2021/09/causal-inference-using-instrumental-variables) used an old A/B test as an instrument to measure the effect of Avatar Shop engagement on community engagement. The test, originally designed to evaluate a recommendation feature, strongly influenced shop engagement but had no other impact on the platform—satisfying both relevance and exclusion.

- [At Booking.com](https://booking.ai/encouraged-to-comply-3cd95b447a20), Account Managers (AMs) attempt to reactivate property partners who have left the platform. To measure the effect of these reactivation calls, a randomized encouragement design is used: some partners are randomly assigned to be contacted (instrument), but AMs retain discretion over whether to actually make the call (treatment). This setup introduces noncompliance, which prevents the use of traditional A/B testing to identify the causal effect of the call. Instead, the company applies the Instrumental Variable approach to estimate bounds on the Average Treatment Effect (ATE), relying on the random assignment as a valid instrument. This method accounts for the reality that AMs do not always follow assignments, while still enabling Booking.com to draw meaningful, data-driven conclusions about the value of outreach efforts.

- Another common example is using an experiment as an instrument to correct for non-compliance. Suppose the treatment is a pop-up banner designed to encourage conversion, but on some devices it doesn't appear due to ad blockers. This creates non-compliance: some users assigned to treatment don’t actually receive it. The estimated treatment effect (intent-to-treat effect) will be attenuated (biased towards 0), since it includes users who didn’t receive the treatment. By using treatment assignment as the instrument and actual treatment received as the treatment variable, we can recover the local average treatment effect (LATE) for compliers: those whose treatment status was influenced by assignment (e.g. users without ad blockers). This addresses the potential confounding of treatment received (e.g. if tech-savvy or higher-income users are more likely to block ads). If non-compliance is actually random, LATE should generalize to the full population, and we recover the average treatment effect (ATE). A variation of this example is when you over-expose an experiment, and want to measure the effect only on a subset of the population.

### Multiple instruments

A core limitation of IV analysis is that it estimates the causal effect for the compliers (LATE), i.e. individuals whose treatment status is influenced by the instrument Z. Since different instruments influence different subsets of the population, LATEs can vary across studies or contexts. This raises an important question: how representative is a given LATE of the true Average Treatment Effect (ATE) for the broader population? One strategy to address this is to use multiple, independent instruments. Each instrument identifies a LATE for its own set of compliers. By collecting these localized causal effects, we can explore how treatment effects vary across subgroups, and, under certain conditions, approximate the ATE by combining them.

Two main approaches:
**Joint model (multiple instruments together)**: Include all instruments simultaneously in the model. This produces a weighted average of the LATEs, where weights reflect each instrument’s first-stage strength (i.e. how strongly it shifts the treatment). All instruments must be valid (satisfying relevance and exclusion). For example, Angrist & Krueger (1991) combined quarter of birth (affects students on the margin of school-entry age) and compulsory schooling laws (affects those who would have dropped out without the law) to instrument for education, leveraging variation in both school entry and legal leaving age.
- **Separate models (one per instrument)**: Estimate separate IV models using one instrument at a time. This allows more flexibility in reporting heterogeneity in treatment effects across subpopulations. You can then combine the estimates using weighted averages. Weights might reflect sample size, inverse variance (as in meta-analysis), or instrument strength (e.g. Cov(T,Z)). For example, you might use several independent past experiments as instruments.

However, averaging multiple LATEs doesn’t guarantee you recover the ATE. As an analogy: each LATE is like a torch illuminating one part of a room. To reconstruct the full picture (the ATE), you either need to shine lights in the right places or understand the room’s layout. You can only recover the ATE if one of the following is true:
- Treatment effects are homogeneous: If the treatment has the same effect for everyone, then the LATE equals the ATE, no matter who the compliers are; or
- Compliers are representative of the full population: If the compliers have the same distribution of treatment effects as the overall population, then even if effects vary, the average effect for compliers still matches the ATE; or
- You can model heterogeneity explicitly: If you can model how treatment effects vary across individuals, and know who the compliers are for each instrument, you can "piece together" the ATE using the appropriate weights or adjustments (e.g. using covariates, interaction terms, hierarchical models, or meta-regression).

In practice, compliers often differ systematically. They may be more persuadable, more marginal, or behave differently in ways that affect treatment response. Understanding and adjusting for these differences is essential when trying to generalize from LATEs to broader conclusions.

### Adding Covariates

Additional covariates are often necessary to satisfy identifiability assumptions when using observational instruments (e.g. adding confounders of the instrument Z and the outcome Y to both stages in 2SLS). However, this does not estimate the Local Average Treatment Effect (LATE) conditional on those covariates. Instead, the result is a weighted average of conditional LATEs, where the weights are determined by how the instrument varies across covariate strata and how those strata relate to the outcome. If treatment effects are heterogeneous across strata, or if defiers are present, these weights can become non-intuitive. They may be negative, exceed one, or fail to reflect subgroup sizes or first-stage strength. As a result, even with a valid instrument, the IV estimate $$\beta$$ can be distorted and difficult to interpret as the effect for any specific population subgroup.

To address this, [Słoczynski (2024)](https://arxiv.org/pdf/2011.06695) proposes two remedies:

- **Use multiple instruments** and test for consistency across estimands using overidentification tests (for example, Hansen’s J test). If the instruments are valid and identify the same causal effect, the estimates should agree.
- **Reordered IV**: Remove the influence of covariates by residualizing each variable with respect to X, then perform IV on the residualized variables:

  $$
  \tilde{T} = T - \mathbb{E}[T \mid X]
  $$
  $$
  \tilde{Z} = Z - \mathbb{E}[Z \mid X]
  $$
  $$
  \tilde{Y} = Y - \mathbb{E}[Y \mid X]
  $$

  Then estimate:

  $$
  \tilde{T} = \alpha \tilde{Z} + \varepsilon_T
  $$
  $$
  \tilde{Y} = \gamma \tilde{Z} + \varepsilon_Y
  $$
  $$
  \beta = \frac{\gamma}{\alpha} = \frac{\text{Cov}(\tilde{Y}, \tilde{Z})}{\text{Cov}(\tilde{T}, \tilde{Z})}
  $$

  This approach ensures the final estimate reflects a cleaner average of the conditional LATEs, one where the weights depend only on the distribution of the covariates, not their relationship to the instrument or outcome.

### Bias

Even with a valid instrument, a weak correlation between Z and T leads to noisy and potentially biased IV estimates. When the instrument is weak, the first-stage regression poorly predicts the treatment ($$\hat{T}$$). Although the noise is symmetric, the estimated first-stage coefficient can be close to zero or even change sign. Since the IV estimator can be expressed as $$\hat{\beta} = \frac{Cov(Z,Y)}{Cov(Z,T)}$$, dividing by a small or near-zero denominator causes the estimate to "blow up," producing a few extremely large positive outliers. Sensitivity tests are therefore important, because even small violations of the assumptions can result in a large bias. Importantly, the direction of the bias in IV estimates typically matches that of OLS. For example, if confounders are positively related to the treatment but negatively related to the outcome, both OLS and IV estimates will underestimate the true effect (negative bias).

<img src="/assets/ols_v_iv_bias.png" alt="Bias by method" width="90%"/>

However, IV is consistent, meaning it will approach the population value as the sample increases. Whereas, under confounding, the OLS estimate remains biased:

<img src="/assets/ols_v_iv_bias_by_n.png" alt="Bias by sample size" width="90%"/>

### Deep IV

While most IV applications use Two-Stage Least Squares (2SLS), which assumes linear and homogeneous treatment effects, Deep IV allows for flexible, non-parametric modeling of heterogeneous treatment effects

### Causal strength

The ideal setting for IV is more restrictive than other methods. However, when a strong instrument is available, IV can be a powerful tool. It enables estimation of causal effects even when unmeasured confounding is present, and can also handle issues like simultaneity bias (feedback loop) and measurement error in the treatment (if uncorrelated with the instrument). However, IV relies on a distinct set of assumptions that are largely theoretical and often untestable, so the credibility of any causal claims hinges on their plausibility in the specific context.

- **Exchangeability** (independence and exclusion restriction): Instead of assuming that the treatment T is independent of the potential outcomes Y (which fails in the presence of unmeasured confounders), IV analysis assumes that the instrument Z affects the outcome Y only through its effect on T. That is, Z must be conditionally independent of Y given T (i.e., no direct or confounding paths from Z to Y). This is known as the exclusion restriction and is not testable with data.
- **Positivity** (overlap): The instrument Z must exhibit sufficient variation across the population (i.e. we observe both values of Z) and must have a strong enough effect on the treatment T. This ensures that changes in Z induce observable changes in T, which in turn generate variation in Y. Weak instruments violate this and lead to imprecise or biased estimates.
- **Consistency**: IV assumes well-defined and stable treatments and instruments, and that each unit’s outcome reflects the treatment they actually received. Violations can arise if the instrument operates inconsistently, affects the outcome through unintended channels, or interferes across units.

These assumptions mean IV is best for cases where there's a lot of unmeasured confounding, a binary and time fixed treatment, and a strong instrument. Further, the effect should be homogenous, or you are truly interested in the effect in the compliers, and monotonicity holds.

## Difference in differences

Difference-in-Differences (DiD) is another quasi-experiment technique used to estimate causal effects when we observe outcomes before and after a treatment for both a treated and control group. It is especially useful for macro-level interventions such as policy evaluations (e.g. assessing the effect of a new law on unemployment) or marketing campaigns that lack granular tracking (e.g. TV ads or billboards). It would be a mistake to simply compare outcomes before and after the intervention in the treated group alone, as this ignores other time-varying factors unrelated to treatment that may have affected the outcome (e.g. seasonality, any underlying trend). The control group serves as a counterfactual, showing what would have happened to the treated group had the intervention not occurred. However, we also can't just compare the treated and control group at the post treatment time period either as there can be pre-existing differences. To get around this, we look at the difference between the pre-post difference of each group (cancelling out any pre-existing difference between the groups by comparing groups, and cancelling out any time invariant confounders by comparing time periods). The key assumption is that, in the absence of treatment, both groups would have followed parallel trends. If that's true, then any divergence in outcome trends after treatment can be attributed to the treatment effect. 

Suppose we have two time periods: pre-treatment (t=0) and post-treatment (t=1); and, two groups: treated (D=1) and control (D=0). Using potential outcomes notation, we can write these as $$Y_D(t)$$ (e.g. $$Y_1(0) \mid D=1$$ is the potential outcome under treatment before treatment is implemented, averaged among those who were actually treated). 

Using this notation, the DiD estimator (ATT) is:

$$
\hat{ATT} = \left( \mathbb{E}[Y_1(1) \mid D=1] - \mathbb{E}[Y_0(1) \mid D=0] \right) - \left( \mathbb{E}[Y_1(0) \mid D=1] - \mathbb{E}[Y_0(0) \mid D=0] \right)
$$

Where:  
- $$\mathbb{E}[Y_1(1) \mid D=1]$$ is observed and estimated by $$\overline{Y}_{\text{treated, post}}$$  
- $$\mathbb{E}[Y_0(1) \mid D=0]$$ is the unobserved treated counterfactual, estimated by $$\overline{Y}_{\text{control, post}}$$  
- $$\mathbb{E}[Y_1(0) \mid D=1]$$ is the unobserved control counterfactual, estimated by $$\overline{Y}_{\text{treated, pre}}$$  
- $$\mathbb{E}[Y_0(0) \mid D=0]$$ is observed and estimated by $$\overline{Y}_{\text{control, pre}}$$

Re-arranging the formula to group the treated and control terms, we switch to the observed outcomes to get the Difference-in-Differences (DiD) formula:

$$\widehat{\text{DiD}} = \big(\overline{Y}_{\text{treated, post}} - \overline{Y}_{\text{treated, pre}}\big) - \big(\overline{Y}_{\text{control, post}} - \overline{Y}_{\text{control, pre}}\big)$$

To estimate what would have happened to the treated group if they hadn’t been treated (the counterfactual), we're taking the pre-treatment average for the treated group, and adding the change observed in the control group (since they weren't treated):

$$
\mathbb{E}[Y_0(1) \mid D = 1] = \overline{Y}_{\text{treated, pre}} + \left( \overline{Y}_{\text{control, post}} - \overline{Y}_{\text{control, pre}} \right)
$$

### Regression

While you only need the group averages for the DiD formula above, converting it to a regression allows us to easily compute the standard error and confidence intervals. For this we need granular data (i.e. one row for each unit i and time period t). Using $$\text{Post}_t$$ to indicate the time period (1 if after treatment, 0 if before), and $$\text{Treated}_i$$ to indicate treatment (1 if unit i is in the treated group, 0 otherwise), we get:

$$Y_{it} = \alpha + \beta \cdot \text{Post}_t + \gamma \cdot \text{Treated}_i + \delta \cdot (\text{Treated}_i \times \text{Post}_t) + \varepsilon_{it}$$

Where:
- $$Y_{it}$$: Outcome for unit i at time t
- $$\alpha$$: Intercept (the average outcome for the control group in the pre-treatment period, i.e. when both Post = 0 and Treated = 0).
- $$\beta$$: The average change in outcome for the control group from pre-treatment to post-treatment (i.e. Post=1, Treated=0). This captures time trends common to both groups (e.g. seasonality, inflation).
- $$\gamma$$: The difference in outcomes between the treated and control groups pre-treatment (i.e. Post=0, Treated=1). This reflects any time-invariant baseline differences between the two groups.
- $$\delta$$: The interaction term only equals 1 for the treated group after treatment (i.e. Post=1, Treated=1). Therefore, it estimates how much more (or less) the treated group changed from before to after treatment, relative to the control group (i.e. any extra change in the treated group after controlling for the general time trends and baseline differences). This is the **DiD estimator**, the average treatment effect on the treated (ATT). 
- $$\varepsilon_{it}$$: Error term (unobserved factors)

Coming back to the DiD formula, we're estimating the four group means from the model:

| Group         | Post\_t | Treated\_i | Interaction | Expected Y                         |
| ------------- | ------- | ---------- | ----------- | ---------------------------------- |
| Control, Pre  | 0       | 0          | 0           | $\alpha$                           |
| Control, Post | 1       | 0          | 0           | $\alpha + \beta$                   |
| Treated, Pre  | 0       | 1          | 0           | $\alpha + \gamma$                  |
| Treated, Post | 1       | 1          | 1           | $\alpha + \beta + \gamma + \delta$ |

Then we can do the same subtractions as before to get the DiD estimator:

$$\widehat{\text{DiD}} = \big(\overline{Y}_{\text{treated, post}} - \overline{Y}_{\text{treated, pre}}\big) - \big(\overline{Y}_{\text{control, post}} - \overline{Y}_{\text{control, pre}}\big)$$

To visualise what we're doing, let's create the counterfactual line by adding $$\gamma$$ (i.e. the difference between treated and control at the pre-treatment time point) to the control line. This assumes the growth in the control is the same as the treated would have been had they not received treatment. The gap between the counterfactual and the observed value for the treated at the post-treatment time point is our DiD estimate $$\delta$$ (ATT).  

<img src="/assets/DiD_regression_plot.png" alt="DiD plot" width="70%"/>

To show how this is differencing out unobserved confounders that are constant over time, suppose the true data-generating process is:

$$Y_{it} = \alpha_i + \gamma_t + \delta D_{it} + \varepsilon_{it}$$

Where:
- $$\alpha_i$$ captures all unit-specific effects (e.g. innate characteristics, geography, etc.)
- $$\gamma_t$$ captures time effects common to all units (e.g. inflation, a new law, seasonality)
- $$D_{it}$$ is the treatment indicator (1 if unit $$i$$ is treated at time $$t$$)
- $$\delta$$ is the treatment effect of interest
- $$\varepsilon_{it}$$ is an idiosyncratic error (noise or residual variation that is specific to an individual unit and time, but not systematically related to treatment or time)

By looking at differences over time within units, DiD cancels out $$\alpha_i$$ (the time-invariant characteristics), because they don’t change between periods. Then, by comparing across groups, it cancels out $$\gamma_t$$ (the common time based effects), as long as both groups are equally affected. This leaves just the treatment effect ($$\delta$$) and noise ($$\varepsilon$$). However, given we assume parallel trends in the absence of treatment, it does not remove bias from time-varying confounders that affect the groups differently (e.g. one group improves faster over time regardless of treatment).

### Example

Suppose a new education policy (e.g. a new curriculum) is implemented only in one school zone, beginning in year t=1. You want to evaluate the impact of this policy on student test scores. Students are observed over time in both the treated and control zones.

$$Y_{it} = \alpha_i + \gamma_t + \delta D_{it} + \varepsilon_{it}$$

- **$$Y_{it}$$** = Test score for student $$i$$ at time $$t$$  
- **$$D_{it}$$** = 1 if student $$i$$ is in the treated zone **after** the policy is implemented (i.e., post-treatment), 0 otherwise
- **$$\alpha_i$$**: Captures **student- or school-specific fixed effects** (do not change over time), such as:
  - School zone characteristics (e.g., average income, teacher quality)
  - Baseline differences in student ability or parental education
  - Geographic or demographic differences
- **$$\gamma_t$$**: Captures **year-specific shocks common to all zones**, such as:
  - Changes to the national curriculum or test format
  - Macroeconomic conditions
  - COVID-19 disruptions or weather-related school closures in year $$t$$
- **$$\delta D_{it}$$**: The **causal effect of the education policy**, i.e., the average treatment effect on the treated (ATT) — this tells us how test scores changed **due to the policy** in the treated zone, **above and beyond** what would have happened without the policy.
- **$$\varepsilon_{it}$$**: The **idiosyncratic error term**, capturing random or unobserved student-specific shocks that vary over time:
  - The student had a bad day or was sick during the test
  - A fire drill interrupted the exam
  - Random variation in scoring or guessing

### Extensions

**Covariates**: You can add observed covariates to the DiD model to improve precision or adjust for time-varying confounding (any residual imbalance in characteristics that change over time and differ across groups). This doesn't change the identifying assumptions but helps reduce residual variance (narrower confidence intervals). Below, $$\mathbf{X}_{it}$$ is the vector of covariates and $$\mathbf{\beta_3}$$ are the corresponding coefficients:

$$Y_{it} = \beta_0 + \beta_1 \text{Post}_t + \beta_2 \text{Treated}_i + \delta \big(\text{Treated}_i \times \text{Post}_t\big) + \mathbf{\beta_3} \mathbf{X}_{it} + \epsilon_{it}$$

**Multiple time periods**: The classic DiD setup compares two time points: pre-treatment and post-treatment. But DiD is easily generalized to panel data with multiple time periods. This lets you see how the treatment effect evolves over time — before and after treatment. Instead of one single $$\delta$$, you have a vector of treatment effects $$\delta_\tau$$ indexed by time since treatment:

$$Y_{it} = \alpha_i + \gamma_t + \sum_{\tau \geq 0} \delta_\tau \cdot 1(t = \tau) \cdot D_i + \boldsymbol{\beta} \mathbf{X}_{it} + \epsilon_{it}$$

   - $$\alpha_i$$ and $$\gamma_t$$ are still your fixed effects for units and time.
   - $$D_i$$ is a simple indicator that unit $$i$$ ever receives the treatment (e.g., treated school districts).
   - $$1(t = \tau)$$ is an indicator variable that equals 1 if the current time period $$t$$ is exactly $$\tau$$ periods after the treatment started.
   - The product $$1(t = \tau) \cdot D_i$$ is thus 1 only for treated units at exactly $$\tau$$ periods after treatment.
   - $$\delta_\tau$$ is the treatment effect specific to the $$\tau$$-th post-treatment period.

Say you have data for 3 years after treatment starts ($$\tau$$ =0,1,2): 
- $$\delta_0$$ measures the immediate impact of treatment in the first period.
- $$\delta_1$$ measures the effect in the second period after treatment.
- $$\delta_2$$ measures the effect in the third period after treatment.

By estimating separate $$\delta_\tau$$ for each year, you can see whether the trend. For example it may: peak immediately and decline; build up gradually; remain stable; or behave in some other pattern. This richer model is often called an event study or dynamic DiD because it studies the evolution of treatment effects over time. 

### Assumptions:

**Parallel Trends**:  
The key identifying assumption of Difference-in-Differences (DiD) is that, in the absence of treatment, the average outcome for the treated group and the control group would have followed the same trend over time. Formally, if we let $$Y_{it}(0)$$ be the potential outcome without treatment for unit $$i$$ at time $$t$$, then:

$$
\mathbb{E}[Y_{it}(0) - Y_{i,t-1}(0) \mid \text{treated}] = \mathbb{E}[Y_{it}(0) - Y_{i,t-1}(0) \mid \text{control}]
$$

In other words, we assume we can use the control as the counterfactual, so any differences in outcomes between treated and control groups before the treatment persist but do not change over time. If this assumption fails (e.g. treated group is on a systematically different trajectory), the DiD estimate will be biased. In industry this is often an issue because the choice of who to treat is targeted (e.g. testing a new feature in a lower performing market to try improve it). In practice, we might find a control by plotting pre-treatment trends in the outcome for each potential control vs the target group. If they appear parallel before treatment, the assumption is more credible:

<img src="/assets/DiD_parallel_assumption_plot.png" alt="DiD assumption plot" width="70%"/>

In addition to eyeballing it, we can fit a regression on the pre-treatment time period, using interaction terms (Treated x Time) to assess whether there is a pre-treatment difference in the slopes. Starting with a simple case of two time periods pre-treatment:

$$
Y_{it} = \beta_0 + \beta_1 \, \text{Treated}_i + \beta_2 \, \text{Time}_t + \beta_3 \, (\text{Treated}_i \times \text{Time}_t) + \varepsilon_{it}
$$


| Term                                        | Value                                                      | Coefficient  | Meaning                                                                                                                                                                                                                                                                         |
| ------------------------------------------- | ---------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| \$\text{Treated}\_i\$                       | 1 if unit \$i\$ is treated, 0 otherwise                    | \$\beta\_1\$ | Measures the **average difference in outcomes** between treated and control groups **at time = 0** (the baseline period).                                                                                                                                                                                |
| \$\text{Time}\_t\$                          | 1 if time = 1 (later pre-period), 0 if time = 0 (baseline) | \$\beta\_2\$ | Measures the **average change in outcomes** from time 0 to time 1 for the **control group** — that is, the baseline trend in the absence of treatment.                                                                                                                                                   |
| \$\text{Treated}\_i \times \text{Time}\_t\$ | 1 only if unit is treated **and** time = 1                 | \$\beta\_3\$ | Measures the **difference in trends** from time 0 to 1 between treated and control groups. Specifically, it compares the change for **treated units** to that for **control units**. A non-zero value means the groups had different trends before treatment — violating the parallel trends assumption. |

In this case, we could just use the t-test results for $$\beta_3$$ to assess whether there is a significant difference. However, we can generalise this to multiple time periods and use an F-test to check if any of the interactions are significant:

$$
Y_{it} = \beta_0 + \beta_1 \text{Treated}_i + \sum_{t=1}^T \beta_{2t} \, \text{Time}_t + \sum_{t=1}^T \beta_{3t} \, \big(\text{Treated}_i \times \text{Time}_t\big) + \varepsilon_{it}
$$

Where:
- $$\text{Treated}_i = 1$$ if unit $$i$$ is treated, 0 otherwise,
- $$\text{Time}_t$$ are dummy variables for each time period $$t$$, with the base period $$t = 0$$ omitted,
- $$\text{Treated}_i \times \text{Time}_t$$ are the interaction terms for treated units in time period $$t$$,
- $$\beta_{3t}$$ coefficients test whether the treated group’s trend differs from control’s in period $$t$$ relative to the base period.

In Python, after identifying the names of the interactions, you can write the hypothesis ($$H_0: \beta_{31} = \beta_{32} = \cdots = \beta_{3T} = 0$$) as a string and pass it to f_test in statsmodels (to restrict the test to only the interactions). If it's significant, this indicates a violation of the assumption. If there are many pre-treatment time periods, one option is to aggregate them, or pick certain lags (e.g. -1 month, -2 months). However, keep in mind this is only checking if they were parrallel pre-treatment and doesn't rule out a difference in the post-treatment time period that is unrelated to treatment.

**No Spillovers (Stable Unit Treatment Value Assumption - SUTVA)**:  
The treatment assigned to one unit (e.g. one school zone) does not affect the outcomes of units in the control group. For example, if the new education policy in the treated zone indirectly improves resources or student outcomes in the control zone, this assumption is violated, potentially biasing estimates. This is harder to assess directly from the data, and relies instead on being familiar with the intervention and outcomes.

**No Composition Change (No Differential Attrition or Migration)**:  
The composition of the treated and control groups remains stable over time, and changes in group membership are not related to the treatment. For example, if after the policy starts, higher-performing students move into the treated zone (or control zone students move away), the comparison groups are no longer comparable, and the DiD estimator may be biased. If you have stable IDs over time, you can catch this by checking if any units are assigned to multiple treatments.

### Causal strength

DiD is not suitable for all cases, but offers a simple way to control for unobserved, time-invariant confounders and common shocks when we have panel data (measurements over time), and the assumptions are plausible. 

- **Exchangeability** (parallel trends): In DiD, instead of assuming that the treatment D is independent of the potential outcomes Y (which fails in the presence of unmeasured confounders), DiD assumes that, in the absence of treatment, the treated and control groups would have followed parallel trends (i.e. the growth in potential outcomes is independent of treatment assignment). This weaker assumption allows for unobserved confounding, as long as it is constant over time and affects both groups similarly (so it will get cancelled out). 
- **Positivity** (overlap): Each group must have a positive probability of being observed both before and after the intervention. This means the groups must be present and measurable in all relevant time periods—otherwise we cannot estimate the group-specific trends needed for DiD.
- **Consistency**: The observed outcome for a unit under the treatment received and time period observed reflects the potential outcome for that treatment-time combination. Meaning: we are seeing the correct potential outcome (SUTVA); we are seeing it for a stable set of units (no composition change); and, we’re measuring it accurately, without bias introduced by how outcomes are recorded or constructed.

DiD is generally best for situations where:
- Treatment is assigned at a group level or policy is rolled out at a known time.
- There is reliable longitudinal data on treated and control groups.
- The parallel trends assumption is plausible or can be tested with pre-treatment data.
- Unobserved confounding is largely time-invariant.

## Two-Way Fixed Effects (TWFE)

TWFE is essentially a regression implementation of DiD that generalizes to more than two time periods, staggered treatment timing (units get treated at different times), and continuous treatment variables. By adding the unit itself as a control to the model (i.e. a binary variable for each ID), we control for all of the things that are fixed for a unit even if we can't measure them (e.g. over a short time period a person's IQ is likely unchanged).

The TWFE model is typically written as:

$$
Y_{it} = \alpha_i + \lambda_t + \tau D_{it} + \varepsilon_{it}
$$

Where:

- $$Y_{it}$$: Outcome for unit $$i$$ at time $$t$$  
- $$\alpha_i$$: Unit fixed effect, controls for time-invariant characteristics of each unit  
- $$\lambda_t$$: Time fixed effect, controls for shocks common to all units in period $$t$$  
- $$D_{it}$$: Treatment indicator, 1 if unit $$i$$ is treated at time $$t$$, 0 otherwise  
- $$\tau$$: The treatment effect (what we aim to estimate)  
- $$\varepsilon_{it}$$: Error term capturing unexplained variation  

When estimating this model in a regression, you don’t estimate $$\alpha_i$$ and $$\lambda_t$$ directly. Instead, they are encoded as dummy variables, dropping one to avoid perfect multicollinearity: 

- $$\alpha_i$$: For N units, you include n-1 dummy variables (unit2, unit3, ..., unitN). These dummies capture any unit-specific, time-invariant differences in outcome levels (e.g. permanent differences across individuals or regions).
- $$\lambda_t$$: For T time periods, you include t-1 dummy variables (time2, time3, ..., timeT). These capture any period-specific shocks or trends that affect all units equally (e.g. economic trends or policy changes affecting everyone).

The resulting regression looks something like this:

$$
Y_{it} = \tau D_{it} + \sum_{i=2}^{N} \beta_i \cdot \text{unit}_i + \sum_{t=2}^{T} \gamma_t \cdot \text{time}_t + \varepsilon_{it}
$$

### Demeaning (Within Transformation)

In practice, to avoid adding n-1 dummy variables which could be very large, we can use a method called "demeaning". When you include dummy variables in a regression, you're estimating and removing their average effect (or conditional average if there's additional covariates). De-meaning does exactly the same thing, but more efficiently: it subtracts the averages directly, without estimating a coefficient for each dummy.

**Step 1: Compute Averages**

- **Unit (i) averages:**

$$
\bar{Y}_i = \frac{1}{T_i} \sum_t Y_{it}, \quad \bar{D}_i = \frac{1}{T_i} \sum_t D_{it}
$$

- **Time (t) averages:**

$$
\bar{Y}_t = \frac{1}{N_t} \sum_i Y_{it}, \quad \bar{D}_t = \frac{1}{N_t} \sum_i D_{it}
$$

- **Overall averages:**

$$
\bar{Y} = \frac{1}{NT} \sum_{i,t} Y_{it}, \quad \bar{D} = \frac{1}{NT} \sum_{i,t} D_{it}
$$

**Step 2: Two-Way Demeaning**

Create demeaned variables by subtracting the averages, which removes both $$\alpha_i$$ and $$\lambda_t$$. Regressing on unit dummies controls for unit averages, so subtracting the unit means is a manual but more efficient way of removing the effect:

- **Demeaned outcome:**
$$
\tilde{Y}_{it} = Y_{it} - \bar{Y}_i - \bar{Y}_t + \bar{Y}
$$

- **Demeaned treatment:**
$$
\tilde{D}_{it} = D_{it} - \bar{D}_i - \bar{D}_t + \bar{D}
$$

**Step 3: Regress the demeaned outcome on the demeaned treatment** 

The de-meaning process yields the same estimate as running a full regression with one-hot dummies for each unit and time period:

$$
\tilde{Y}_{it} = \tau \tilde{D}_{it} + \tilde{\varepsilon}_{it}
$$

| Component       | Explanation                                                            |
|----------------|------------------------------------------------------------------------|
| $$\alpha_i$$    | Removed via subtracting unit average                                   |
| $$\lambda_t$$   | Removed via subtracting time average                                   |
| $$\tau$$        | Estimated by regressing demeaned $$Y_{it}$$ on $$D_{it}$$              |

### Assumptions

There are the functional form assumptions:
- Treatment Effect Heterogeneity: If the treatment effect varies across time or units, the TWFE estimate τ can be a weighted average of different effects, with some weights being negative. This can lead to misleading conclusions, as shown in recent research (e.g., Goodman-Bacon decomposition).
- Linearity in the covariates;
- Additive fixed effects.

But also the strict exogeneity assumptions:
- Parallel Trends: In the absence of treatment, treated and control units would have followed the same trend over time. TWFE enforces this assumption implicitly through fixed effects.
- No time-varying confounders: TWFE can't handle variables that change over time and differ systematically between treated and control units — unless you explicitly model those.
- No anticipation
- Past treatment don’t affect current outcome (no carryover)
- Past outcome don’t affect current treatment (no feedback)

### Bias

Recent econometrics research showed that staggered DiD with TWFE is biased when treatment effects are heterogeneous (post-2018, notably Goodman-Bacon, Sun & Abraham, Callaway & Sant’Anna). This is especially problematic when different groups adopt treatment at different times (e.g., state policies rolled out in different years), or when treatment effects grow or shrink over time. These patterns are common in real-world interventions, for example, an e-commerce site rolling out a feature by country, with the effect of the feature taking time to fully materialize.

The core problem is that some units act as controls for others after they’ve already been treated, causing TWFE to assign negative weights to certain comparisons. If the treatment effect increases over time, this tends to bias the overall estimate downward; if the effect decreases over time, it biases the estimate upward. As a result, the TWFE estimate isn’t just a weighted average of the group-specific effects, it’s a distorted mix of comparisons that can even have the wrong sign if treatment effects vary enough.

<img src="/assets/TWFE_stagger.png" alt="Stagger example" width="70%"/>

For staggered rollouts, researchers have increasingly moved toward more robust alternatives that explicitly model treatment effect heterogeneity, such as:
- Callaway & Sant’Anna (2021): Group-time average treatment effects.
- Sun & Abraham (2021): Interaction-weighted estimators correcting the TWFE bias.
- Gardner (2022) Did2s: A two-stage procedure that improves robustness in TWFE designs.

Using these approaches allows you to estimate group-specific treatment effects over time and avoid the misleading averages produced by naïve TWFE models in heterogeneous settings.

### Causal strength

TWFE models extend DiD to more flexible settings with multiple time periods and variation in treatment timing. They offer a simple way to control for unobserved time-invariant confounders (via unit fixed effects) and time shocks common across units (via time fixed effects), using a linear regression framework.

- **Exchangeability** (parallel trends, conditional on fixed effects): Instead of assuming that treatment $$D$$ is independent of potential outcomes $$Y$$, TWFE relies on the assumption that, *after removing unit and time fixed effects*, treatment assignment is as good as random. In other words, in the absence of treatment, each unit’s outcome would have evolved similarly to others after accounting for fixed differences. This assumption allows for time-invariant unobserved confounding, as long as treatment timing is not systematically related to time-varying shocks.
- **Positivity** (overlap): There must be sufficient variation in treatment timing across units and time periods. That is, some units must be untreated while others are treated in each period (or across periods), so that treatment effects can be compared within the model. Without overlap, the model cannot identify treatment effects separately from unit or time effects.
- **Consistency**: The observed outcome $$Y_{it}$$ equals the potential outcome $$Y_i(D_{it}, t)$$. This requires:
  - SUTVA (no interference between units, and a well-defined treatment),
  - No composition change over time (units remain comparable),
  - Reliable measurement of outcomes, with no differential measurement error by treatment or time.

TWFE is especially useful when:
- You have panel data with many time periods and staggered treatment adoption.
- You believe fixed differences between units and common time shocks are key confounders.
- Treatment varies within units over time (within-unit comparisons are possible).
- You want to estimate an average effect across units and time, accounting for these fixed effects.

## Synthetic controls

Synthetic controls overcome the problem of finding a single good control for a DiD model. Instead of selecting one comparison unit, we take a weighted average of multiple control units to create a synthetic control that closely matches the treated unit’s trend during the pre-intervention period.

For example, imagine an e-commerce company runs a test for a new product in 1 of the 100 cities it operates in. Instead of choosing one comparison city, which may still not perfectly match the treated city, we construct a weighted combination of the remaining 99 cities to form a synthetic version. We then use this synthetic control as a proxy for what would have happened to the treated city post-intervention if it had not been treated.

### Synthetic outcome formula

$$
\hat{Y}^N_{1t} = \sum_{j=2}^{J+1} w_j Y_{jt}
$$

Where:

- $$\hat{Y}^N_{1t}$$ is the synthetic (estimated) outcome for treated unit 1 at time $$t$$
- $$Y_{jt}$$ is the outcome for control unit $$j$$ at time $$t$$
- $$w_j$$ is the weight assigned to control unit $$j$$
- $$\sum_{j=2}^{J+1} w_j = 1$$ means the weights sum to 1 (convex combination)
- $$w_j \ge 0$$ means the weights are non-negative

### Finding the weights

The weights $$w_j$$ are chosen to minimize the difference between the pre-treatment outcomes of the treated unit and its synthetic counterpart:

$$
\min_{w} \sum_{t=1}^T \left( Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt} \right)^2
$$

Where:

- $$Y_{1t}$$ is the outcome for the treated unit at time $$t$$ (pre-treatment)
- $$T$$ is the number of pre-treatment periods

We can solve for these weights by running a regression where the units (for example, cities) are the features (columns) and the time periods are the observations (rows). However, this requires constrained least squares with:

- $$\sum_{j=2}^{J+1} w_j = 1$$
- $$w_j \ge 0$$ for all $$j$$

These constraints ensure that the synthetic control stays within the range of observed data and avoids extrapolation.

### Assessing significance: Placebo tests

Since we typically have few units (small sample size), standard errors from regression are unreliable. Instead, we use placebo tests:

- For each control unit, pretend it was treated, fit a synthetic control, and compute the resulting placebo effect.
- Compare the gap for the actual treated unit to the distribution of placebo gaps.
- The p-value is the proportion of placebo gaps that are larger than the estimated treatment effect.

This non-parametric test provides a practical way to assess whether the estimated treatment effect is meaningful or could plausibly have arisen by chance.

## Bayesian structural time series

Another approach to estimating the effect of an intervention on time series data is Bayesian structural time series (BSTS), popularized by Google’s `CausalImpact` package.

Where synthetic controls build a weighted average of other units to form a comparison, BSTS builds a model-based forecast of what would have happened to the treated unit if the intervention had not occurred. It incorporates:

- Trends and seasonal patterns, capturing gradual movements or recurring cycles
- Covariates, control variables that help predict the outcome
- Uncertainty, using Bayesian inference to provide credible intervals for the estimated effects

The estimated effect of the intervention is calculated as the difference between the observed data and the predicted counterfactual in the post-treatment period.

### Model structure

A typical BSTS model looks like this:

$$
Y_t = \mu_t + \tau_t + \beta X_t + \epsilon_t
$$

Where:

- $$\mu_t$$ is the local level, capturing gradual underlying trends
- $$\tau_t$$ is the seasonal component, if applicable
- $$X_t$$ represents observed covariates, such as control markets
- $$\beta$$ are the coefficients for each covariate, estimated using spike-and-slab priors for automatic selection and regularization
- $$\epsilon_t$$ is Gaussian random noise

The spike-and-slab priors are especially useful when there are many potential control variables. The spike controls whether a variable is likely to be included, while the slab shrinks included variables toward zero to guard against overfitting and multicollinearity.

### How it works

1. Fit the model on the pre-intervention period to learn patterns in the time series.
2. Forecast the counterfactual to generate predictions for the post-intervention period, representing what would have happened without the intervention.
3. Compare the observed data to the predicted counterfactual to estimate the causal impact.
4. Quantify uncertainty using Bayesian credible intervals derived from the posterior distribution of the forecast.

### Pre-selecting the control pool

While `CausalImpact` uses spike-and-slab priors to automatically select the most predictive control markets for the structural time series model, if you have many potential controls it can be useful to first reduce the pool to more viable candidates. Pre-selecting the control pool improves computational efficiency, reduces noise from unrelated markets, and can help avoid issues with multicollinearity between highly correlated controls.

One practical approach is to select control markets that have demonstrated similar trends to the treated market during the pre-intervention period. Similarity can be measured using methods like:

- Correlation or R-squared between time series
- Dynamic Time Warping (DTW), which allows for small temporal misalignments
- Distance metrics on smoothed or detrended series

The MarketMatching package, developed by Stitch Fix, provides a convenient implementation of this idea. It uses DTW to score similarity between time series and select a subset of controls with trends closest to the treated unit. These selected markets can then be used as covariates in the `CausalImpact` model.

After pre-selecting controls, it’s still important to validate the model by testing its performance on placebo interventions in the pre-treatment period. This helps ensure that the final model is well-calibrated and reduces the risk of detecting spurious effects.

### Interpreting the results

By default, `CausalImpact` produces a three-panel plot:

1. Observed versus predicted, showing the actual series against the counterfactual forecast
2. Pointwise impact, the estimated effect at each time point
3. Cumulative impact, the running total of the effect over time

<img src="/assets/BSTS_plot.png" alt="CausalImpact plot" width="90%"/>

Cumulative impact is most meaningful when the outcome represents a flow, such as daily sales. For metrics defined at a point in time, like the number of customers, the pointwise impact is often more interpretable.

### Benefits

- Models rich time series dynamics, including trends, seasonality, and covariates
- Provides uncertainty intervals for effect estimates
- Supports automatic variable selection when many potential controls are available
- Well-suited for cases with a single treated unit and a long pre-intervention time series

### Limitations

- Relies on the pre-intervention model being well specified; at least 1.5 years of pre-period data is recommended for robustness, especially if cross-validation is used
- Structural breaks or confounding changes after the intervention can bias estimates. Limiting the post-treatment period, for example to one month, can reduce this risk
- Requires more modeling choices than simpler approaches like Difference-in-Differences or synthetic controls
- Frequentist inference is not directly available. Statistical significance is assessed using Bayesian posterior probabilities instead of p-values

### Testing robustness

To assess robustness, placebo tests can be used:

- Simulate “fake” interventions during the pre-treatment period
- Fit the model and check how often it detects a significant effect when no intervention actually occurred
- Use this to estimate the model’s false positive rate and tune parameters like the number of control variables and prior assumptions

### Choosing between DiD / synthetic control / BSTS

| Method                         | Best for                                             | Key assumptions                                 | Strengths                                  | Limitations                                 |
|--------------------------------|-----------------------------------------------------|------------------------------------------------|--------------------------------------------|---------------------------------------------|
| **Difference-in-Differences (DiD)** | Many treated & control units with parallel trends    | Parallel trends (or conditional on covariates) | Simple to implement, interpretable        | Sensitive to trend differences, no flexible control selection |
| **Synthetic Control**          | One treated unit, good set of comparison units      | Treated can be approximated by weighted avg. of controls | Transparent, intuitive counterfactual     | Requires good controls, limited flexibility in dynamics      |
| **Bayesian Structural Time Series (BSTS)** | One treated unit, complex or noisy time series       | Pre-period model well specified, stable covariate relationships | Rich time series dynamics, automatic control selection | More complex setup, relies on prior choices, sensitive to model misspecification |

**Recommendation**:  
- Use **DiD** for larger panel data with multiple treated units.  
- Use **Synthetic Control** when you have few treated units but good controls.  
- Use **BSTS (CausalImpact)** when controls are noisy, or the treated series has trends and seasonality that need to be modeled explicitly.

## Regression Discontinuity

Regression Discontinuity Design is a quasi-experimental method used for estimating causal effects when treatment is assigned based on a cutoff in a continuous variable (called the running variable or forcing variable). Given we're looking at a limited time period, we assume that everything else affecting y changes smoothly at the the cutoff (continuity assumption), so any jump must be due to treatment. For example, if we decide anyone who scores above 80 on a test gets some reward, those who scored 80 are likely very similar to those who scored 81. Graphically, RDD looks like two regression lines, one on each side of the cutoff. The discontinuity (the jump) between those two lines at the cutoff is what we interpret as the causal effect of the treatment.

After centering the running variable, we can specify the Regression Discontinuity Design (RDD) model as:

$$
Y_i = \alpha_0 + \alpha_1 \cdot \tilde{r}_i + \alpha_2 \cdot 1\{\tilde{r}_i > 0\} + \alpha_3 \cdot 1\{\tilde{r}_i > 0\} \cdot \tilde{r}_i + \varepsilon_i
$$

Where:
- $$Y_i$$: Outcome for individual $$i$$  
- $$\tilde{r}_i = r_i - c$$: Running variable centered at the cutoff $$c$$, so that $$\tilde{r}_i = 0$$ at the cutoff. This has 2 advantages: the intercept $$\alpha_0$$ directly represents the expected outcome at the cutoff for the untreated group; and, the treatment effect $$\alpha_2$$ is then interpreted as the causal effect of crossing the threshold.
- $$1\{\tilde{r}_i > 0\}$$: Dummy variable, 1 if individual $$i$$ is **above the cutoff** (treated), 0 otherwise 
- $$1\{\tilde{r}_i > 0\} \cdot \tilde{r}_i$$: Interaction term for the running variable and the cutoff dummy (allowing the slope to differ on either side of the cut-off)
- $$\varepsilon_i$$: Error term

| Coefficient | Measures                             | Interpretation                                 |
| ----------- | ------------------------------------ | ---------------------------------------------- |
| $\alpha_0$  | **Baseline** (untreated)             | Expected outcome at cutoff (coming from below) |
| $\alpha_1$  | **Pre-cutoff slope**                 | Trend of outcome as $r_i \to c^-$              |
| $\alpha_2$  | **Treatment effect at cutoff**       | Causal estimate: jump when we switch on the dummy ($$\lim_{r_i \to c^+} \mathbb{E}[y_i \mid r_i] - \lim_{r_i \to c^-} \mathbb{E}[y_i \mid r_i]$$)         |
| $\alpha_3$  | **Change in slope due to treatment** | Difference in slopes (change when the cutoff dummy switches to 1) |

With our running variable (centered at 0), cutoff dummy, and their interaction, we're able to fit the pre and post cutoff slopes:

<img src="/assets/RDD_plot.png" alt="RDD plot" width="70%"/>

This can also be combined with earlier methods:

| **RD Alone**                                                                        | **RD + DiD (RDDiD)**                                                                                         | **RD + Matching/Covariates**                                                           |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Controls for confounding *at the cutoff* only.                                    | Adds control for **time trends** in **panel or repeated cross-section**.                                     | Improves covariate balance, reduces residual variance.                                 |
| Local effect at the threshold.                                                    | Local **difference in trends** at the threshold.                                                             | Local effect, but more efficient.                                                      |
| You believe continuity in potential outcomes holds and no time trend confounding is likely. | There’s a **policy or event at a specific time** that might affect the groups **even without treatment**.     | You see **imbalance in covariates near the cutoff**, especially with small samples.    |

### Diagnostics / improvements

Given we're essentially fitting two regression slopes, it's possible the model will focus too much on fitting all the points and not on the points at the cutoff (which is what our causal estimate measures). One solution to this is to give more weight to the points near the cutoff, emphasizing the part of the data most relevant for identifying the discontinuity. There are a few ways to do this:

- Kernel weighting: Use a kernel function (e.g., triangular, Epanechnikov) to assign higher weights to observations closer to the cutoff and lower weights to those further away. The triangular kernel is commonly used because it prioritizes points nearest the threshold.
- Bandwidth selection: The choice of bandwidth (how wide a window of data to use around the cutoff) is crucial. A smaller bandwidth reduces bias (focusing on near-cutoff observations) but increases variance (fewer data points). Larger bandwidths reduce variance but risk bias if the relationship between the running variable and the outcome is nonlinear. Data-driven methods (e.g., cross-validation, Imbens-Kalyanaraman) help select optimal bandwidths.
- Higher-order polynomials or local polynomial regression: Instead of simple linear models on either side of the cutoff, you can use quadratic or cubic terms, or fit local polynomial regressions, to better approximate the functional relationship within the chosen bandwidth. Care must be taken—overly flexible models can reintroduce bias by fitting noise.
- Covariate adjustment: While not required for identification, adding pre-treatment covariates can reduce residual variance and improve precision if these covariates are smoothly related to the running variable.

It’s good practice to try multiple bandwidths, kernels, and model specifications to check the sensitivity of your results.

#### McCrary Test

If units can precisely manipulate their position relative to the cutoff (e.g., by sorting or gaming eligibility), the assumption of continuity in potential outcomes may be violated, threatening identification. The McCrary test is a diagnostic for checking manipulation of the running variable around the cutoff. It tests whether there is a discontinuity in the density (distribution) of the running variable at the threshold by comparing densities just to the left and right of the cutoff. A significant jump suggests potential manipulation or sorting, while a smooth density supports the assumption that units cannot manipulate assignment precisely, strengthening the credibility of the RDD.

### Causal strength

Under mild conditions (continuity of potential outcomes at cutoff), RD is as good as randomization near the threshold, making it very convincing when the assumptions hold. However, it only estimates the local average treatment effect (LATE) around the cutoff, not global effects (i.e. it doesn't necessarily generalize to samples far from the cutoff).

- **Exchangeability**: Assumed to hold at the cutoff, because units just above and below the threshold are expected to be similar in both observed and unobserved characteristics. This hinges on the continuity of potential outcomes at the threshold.
- **Positivity (overlap)**: Automatically satisfied near the cutoff, since we observe units on both sides of the threshold by design. However, treatment assignment is deterministic based on the running variable, meaning there’s local overlap, not global.
- **Consistency**: Requires that the treatment actually corresponds to the defined intervention at the threshold, and that there’s no interference between units (one unit’s treatment doesn’t affect another’s outcome).

### Fuzzy RDD (non-compliance)

Standard (sharp) Regression Discontinuity Design (RDD) can be viewed as a special case of an Instrumental Variables (IV) design with perfect compliance at the cutoff. However, in many practical situations, treatment assignment is not perfectly determined by crossing the cutoff. Instead, the probability of receiving treatment increases discontinuously at the threshold, but some units above the cutoff might not receive treatment, and some below might receive it. This setting is known as fuzzy regression discontinuity design and can be estimated using an IV approach, where crossing the cutoff acts as an instrument for actual treatment receipt. The causal effect identified in fuzzy RDD is the local average treatment effect (LATE) for compliers, units whose treatment status is influenced by whether they cross the threshold.

- Let $$Z_i = \mathbf{1}(r_i \geq c)$$ denote the instrument, indicating if the running variable $$r_i$$ is above the cutoff $$c$$.
- Let $$D_i$$ be the actual treatment received by unit $$i$$, which may not perfectly correspond to $$Z_i$$ due to non-compliance or imperfect assignment.
- Let $$Y_i$$ be the observed outcome for unit $$i$$.

Because $$Z_i$$ influences $$D_i$$ only probabilistically, we use $$Z_i$$ as an instrument for $$D_i$$ to estimate the causal effect of treatment on the outcome:

**First stage:** Model the effect of the instrument on treatment assignment:

$$
D_i = \pi_0 + \pi_1 Z_i + f(r_i) + \epsilon_i^D
$$

where $$f(r_i)$$ flexibly models the relationship between the running variable and treatment (for example, separate slopes or polynomial terms on each side of the cutoff).

**Second stage:** Model the outcome as a function of predicted treatment:

$$
Y_i = \beta_0 + \beta_1 \hat{D}_i + g(r_i) + \epsilon_i^Y
$$

where $$\hat{D}_i$$ is the fitted value of treatment from the first stage, and $$g(r_i)$$ models the running variable in the outcome equation.

Since $$Z_i$$ induces variation in $$D_i$$ only near the cutoff $$c$$, the 2SLS estimate $$\hat{\beta}_1$$ recovers the local average treatment effect (LATE) for compliers at the threshold. This IV approach corrects for imperfect compliance, making fuzzy RDD a localized instrumental variables design.

#### Key assumptions:

- Relevance: The instrument $$Z_i$$ must have a strong, discontinuous effect on treatment $$D_i$$ near the cutoff.
- Continuity: Potential outcomes and other confounding factors must be continuous in $$r_i$$ at the cutoff.
- Exclusion restriction: The instrument $$Z_i$$ affects the outcome $$Y_i$$ only through its effect on treatment $$D_i$$ (no direct effect of crossing the threshold on $$Y_i$$).

## Orthogonal Machine Learning

Orthogonal Machine Learning (for example, Double Machine Learning) is a method designed to improve the estimation of causal effects when using flexible machine learning models, particularly in the presence of high-dimensional covariates. It addresses the challenge of nuisance parameters, which are variables that influence the treatment and outcome but are not of primary interest, by orthogonalizing them. This procedure isolates the causal parameter of interest, in a manner conceptually similar to how linear regression isolates specific coefficients after adjusting for other variables.

### Orthogonalization Procedure

The fundamental idea is to decompose both the treatment and the outcome into components that are orthogonal, or uncorrelated, with respect to the nuisance parameters. This is achieved through a two-step process:

1. **Predict Treatment**: Fit a model $$M_t(X)$$ to predict the treatment $$T$$ from the covariates $$X$$. The residuals $$\tilde{t} = T - M_t(X)$$ represent the portion of $$T$$ that is unrelated to $$X$$.

2. **Predict Outcome**: Fit a model $$M_y(X)$$ to predict the outcome $$Y$$ from $$X$$. The residuals $$\tilde{y} = Y - M_y(X)$$ represent the portion of $$Y$$ that is unrelated to $$X$$.

The causal effect is then estimated by regressing $$\tilde{y}$$ on $$\tilde{t}$$, ensuring that the estimated effect reflects the part of $$Y$$ that is attributable to $$T$$ after adjusting for $$X$$.

### Rationale and Properties

A key strength of this approach is that small errors in estimating the nuisance parameters do not bias the estimated treatment effect, provided those errors are orthogonal to the treatment residuals. This robustness is what makes orthogonalization attractive for combining machine learning with causal inference.

To avoid overfitting and to ensure valid confidence intervals, it is recommended to use cross-fitting. This involves training the nuisance models on one subset of the data and computing residuals on a separate subset.

### Advantages

- Robustness to estimation errors in the nuisance models, provided those errors are small
- Flexibility to use sophisticated machine learning models in nuisance estimation while preserving valid inference for the causal parameter
- Valid statistical inference: estimates remain consistent and asymptotically normal, supporting confidence intervals and hypothesis testing

### Limitations

- Unmeasured confounding: Orthogonal Machine Learning assumes that all confounders are observed and included in $$X$$. If important confounders are omitted, estimates will remain biased.
- No resolution of identification problems: This method cannot address issues such as hidden confounding or violations of the overlap assumption between treatment groups.
- Dependence on nuisance model quality: While small estimation errors are acceptable, substantial misspecification in nuisance models will impair the quality of causal estimates.
- Challenges with small samples: Theoretical guarantees assume large samples. In practice, performance may degrade with small or noisy datasets.

### Summary

Orthogonal Machine Learning extends the familiar idea of adjusting for covariates in linear regression to more flexible settings with machine learning models. By orthogonalizing nuisance influences, it isolates the causal effect, providing more accurate and reliable estimation in high-dimensional and complex data contexts. However, as with any causal inference method, its reliability depends on careful model specification, high-quality data, and the plausibility of underlying assumptions.

---

# Evaluating ATE Models

There’s no definitive test to confirm whether a causal estimate is correct, because causal inference always involves estimating unobservable quantities. We only ever see one potential outcome per individual, so we can’t directly check our work. Most methods therefore rely on untestable assumptions, like no unmeasured confounding. That’s why evaluating causal models is more about building confidence than proving correctness. Alongside a clear causal diagram and careful study design, we can use supporting tools and strategies to assess how reliable our estimates might be.

## Sensitivity Tests

Sensitivity analysis helps test whether your estimates are robust to changes in your modeling choices. The general idea is to intentionally vary aspects of your analysis to see if the result holds up. For example:

- Try removing or adding covariates, especially ones where you’re unsure if they’re confounders, mediators, or colliders. If your estimate swings wildly depending on whether a variable is included, that could be a red flag.
- Run the model on subgroups or with different functional forms (e.g. linear vs nonlinear models).
- Swap in a placebo treatment (something you don’t expect to cause an effect), and see if the model wrongly detects a result.
- Use a negative control outcome (an outcome that shouldn’t be affected by the treatment), and check whether your model predicts a change.

These kinds of tests don’t “prove” that your estimate is right, but they help check whether it’s behaving sensibly given your assumptions.

## Consistency Across Evidence

Another form of validation is triangulating across different sources of evidence. While no single model may be perfect, if multiple analyses using different datasets, methods, or assumptions all point in the same direction, that builds credibility. For example:

- External studies can serve as a benchmark.
- Domain expertise can help check if the size of the effect makes sense.
- Comparing experimental and observational estimates (when both are available) is another good test.

Finally, try actively seeking ways your estimate could be wrong. Think through scenarios where your assumptions fail, or design analyses that would disprove your hypothesis. Testing ways to break your result and failing can build confidence in your conclusions.

---

# Conditional ATE 

Estimating the Average Treatment Effect (ATE) is often just the starting point. In many real-world applications, the treatment effect is not constant across all individuals. Some groups may benefit more than others, and for some, the treatment might even be harmful. Understanding this heterogeneity is critical for targeted policies, personalised recommendations, and fair policies.

## Heterogeneity (effect modifiers)

The average treatment effect (ATE) answers the question of whether we should roll out a treatment to the entire population or not. Whereas, estimating the conditional ATE (CATE) for different segments answers the question of who should we treat. Imagine we could observe Y at different values of T for each observation, then we could estimate their sensitivity to treatment based on the slope of the line ($$\frac{dY}{dT}$$). If the slope is large (high elasticity) they are more responsive to treatment, and if it's small (low elasticity) they're not. We could then group people based on their sensitivity and implement the treatment accordingly. 

Now we typically can't observe the same observation with different treatments as discussed above, but we can estimate this value from a sample. For example, suppose we want to estimate how sensitive demand (Y) is to price (T), i.e. the price elasticity of demand. To do this, we could randomly assign users to see the same product at slightly different prices within a reasonable range, say ±10% of the base price. Fitting a linear regression model we can get the ATE straight from $$\beta_1$$ (average price elasticity):

$$Y_i = \beta_0 + \beta_1 T_i + \epsilon_i$$

$$\frac{dY_i}{dT_i} = \beta_1$$

Adding interaction terms allows the treatment sensitivity (elasticity) to vary for different groups:

$$Y_i = \beta_0 + \beta_1 T_i + \beta_2 X_{i1} + \beta_3 X_{i2} + \beta_4 (T_i \cdot X_{i1}) + \beta_5 (T_i \cdot X_{i2}) + \epsilon_i$$

Where:
- $$X_{i1}$$ is income, and $$T_i \cdot X_{i1}$$ captures how the treatment effect varies with income
- $$X_{i2}$$ is past purchases and $$T_i \cdot X_{i2}$$ captures how the effect varies with purchase history

The Conditional Average Treatment Effect (CATE) for observation i is then the partial derivative of $$Y_i$$ with respect to $$T_i$$, conditional on X (using the interaction terms):

$$\frac{dY_i}{dT_i} = \beta_1 + \beta_4 X_{i1} + \beta_5 X_{i2}$$

Now the marginal effect of T depends on the user’s covariates (income and purchase frequency), allowing you to segment users by elasticity and personalize pricing or treatment accordingly. If we plot Y vs X for each level of T, the lines are no longer parallel because we've allowed the slope to vary based on the levels of X. For example:

<img src="/assets/heterogenous_effect.png" alt="Heterogenous effect example" width="80%"/>

Variables that change the magnitude or direction of the treatment effect across different levels of that variable are called effect modifiers. A surrogate effect modifier is a variable that isn't a true modifier itself, but is correlated with the true effect modifier, and is used as a proxy. These are used when true effect modifier is unobserved or difficult to measure, for example click behaviour could be a surrogate for a latent trait like motivation. Existing literature or user research can be useful for validating surrogates when you don't have the data for the modifier.

### Evaluation

There's no direct test to confirm individual CATE estimates are correct, because we never observe both potential outcomes for any single observation. Individual treatment effects are fundamentally unobservable. However, we can evaluate models of heterogeneity using group-level or out-of-sample checks. The best strategy, when feasible, is to use randomized experimental data for the evaluation. While it may not be possible for model training, running smaller-scale experiments solely for validation is often achievable. Using the randomised treatment, we're able to get a reliable estimate of the ATE for groups of observations by comparing the treated to the control. Practical approaches for this include:

- **Group Average Validation:** Divide the sample into bins (e.g. deciles) based on predicted CATE. Then, compare the observed average outcome difference between treatment and control for each group (ATE). If the model captures heterogeneity well, groups with higher predicted CATE should exhibit larger observed treatment effects (i.e. higher sensitivity). If the treatment effect is similar across the groups, then we're not doing a good job at identifying different sensitivity groups.
- **Cumulative Sensitivity Curve:** Extending the above, we can compute the cumulative sensitivity as we move from the most to least sensitive observations based on the predicated CATE. When we plot the cumulative sensitivity on Y and the % of the sample on the X (ranked by predicated CATE), we ideally want the curve to start high, and then slowly converge to the ATE (as the CATE of the full sample is just the ATE). For comparison, a random model will just fluctuate around the ATE. This tells us how good is the model’s average prediction by rank.
- **Cumulative Gain Curve (uplift):** To get something similar to a ROC curve, we can multiply the cumulative sensitivity by the % of the sample used (between 0-1). Multiplying cumulative sensitivity by the % of data used normalizes the cumulative contribution of treatment effect to the overall impact, telling us how much uplift we've accumulated so far by targeting the top-ranked individuals. The baseline (random model) will be a straight line from 0 to the ATE, and all models will end at the ATE (as it's ATE * 1.0). Useful models will show steep curves early on, meaning the highest-ranked individuals benefit most. Flat or erratic curves suggest poor targeting. 
- **Simulated Decisions:** Simulate policy decisions like "treat the top 20% most responsive" and compare the realized lift to random targeting or baseline strategies. This is the same as comparing different cut-offs in the cumulative sensitivity chart, but links heterogeneous effect estimation directly to business outcomes.
- **Simulation Studies:** Finally, simulation studies using synthetic data where the true CATE is known are another validation approach, mostly used in academic or research settings. These can be helpful during model development to build intuition about estimator behavior under different scenarios, but are generally less applicable in applied settings.

<img src="/assets/heterogenous_effect_evaluation.png" alt="Heterogenous effect evaluation" width="100%"/>
Note: If we use regression to compute the sensitivity, we can also easily add a confidence interval using the standard error

Without randomised data, a common approach is to use cross-validation: train your heterogeneity model on one part of the data, then evaluate how predicted CATEs correspond to observed differences in a holdout set. While this helps us understand whether the results are stable, it doesn't tell us whether we're capturing the causal effect (i.e. there can still be bias). Similarly, model performance metrics tell us how well we're fitting the data. This is useful to avoid overfitting (using cross-validation), but it doesn't correspond with the accuracy of the causal estimate. For example, a strong $$R^2$$ means we've fit the data well, but it doesn't tell us whether we're capturing the causal relationship of interest (i.e. it could be biased due to confounding).

Ultimately, strong model metrics, good group-level correspondence, and favorable uplift curves based on randomised data provide practical confidence that the heterogeneity model is capturing useful variation for decision-making, even if individual CATEs remain uncertain. 

## Meta learners

The example above used linear regression to show how we can reframe predictive modeling, not to predict raw outcome values ($Y$), but to predict how the outcome changes with treatment ($$\frac{dY_i}{dT_i}$$). The challenge for applying more flexible or complex machine learning models is that the ground truth isn't observed for each observation (i.e. we only observe one potential outcome). Yet to apply these models directly, we need to define a target that reflects the Conditional Average Treatment Effect ($$\tau(x)$$) instead of the raw Y values. For binary treatments, one effective solution to this is target transformation (F-learners).

Given:

- Binary treatment $$T_i \in \{0,1\}$$
- Outcome $$Y_i$$
- Treatment assignment probability $$p = \mathbb{E}[T]$$ (e.g. 50% in a randomised trial)
- $$\tau(x)$$ = Conditional Average Treatment Effect (CATE)

We define the transformed outcome as:

$$
\tilde{Y}_i = \frac{Y_i - \bar{Y}}{p(1 - p)} \cdot (T_i - p)
$$

With random assignment (or unconfoundedness), this transformation satisfies:

$$
\mathbb{E}[\tilde{Y} \mid X = x] = \tau(x)
$$

This allows us to train any regression or machine learning model to predict $$\tilde{Y}_i$$, and the model’s predictions will approximate $$\tau(x)$$, the individual-level treatment effect. This approach is called an F-learner, where you transform the outcome and then fit a model to predict it. However, the transformed outcome $$\tilde{Y}_i$ $is often noisy, especially when treatment assignment is highly imbalanced. Therefore, this method works best with large datasets where noise averages out.

This is just one example of a meta learner for CATE estimation, with "meta" refering to the fact they wrap around standard supervised learning models to estimate CATE.

| Learner     | What it Estimates                                | Models Fit                                                                                                                                                                 | Formula for CATE $$\hat{\tau}(x)$$                                                                                 | Pros                                                                                         | Cons                                                                                                       |
|-------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| S-learner   | $$\mathbb{E}[Y \mid X, T]$$                     | Single model $$\hat{f}(X, T)$$ (fit model predicting $$Y$$ using $$X$$ and $$T$$ as features)                                                                             | $$\hat{\tau}(x) = \hat{f}(x, 1) - \hat{f}(x, 0)$$                                                                  | Simple; works with any supervised ML model                                                    | Can underfit heterogeneity, especially if regularization shrinks or drops $$T$$ coefficient                |
| T-learner   | $$\mathbb{E}[Y \mid X=x, T=1]$$ and $$\mathbb{E}[Y \mid X=x, T=0]$$ | Two models: $$\hat{\mu}_1(x) = \mathbb{E}[Y \mid X=x, T=1]$$ and $$\hat{\mu}_0(x) = \mathbb{E}[Y \mid X=x, T=0]$$                                                          | $$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$                                                                | Flexible; allows different complexity for treatment/control groups                             | Needs sufficient $$n$$ in both groups; imbalance can distort differences                                    |
| X-learner   | Weighted combination of imputed individual treatment effects | 1. Fit $$\hat{\mu}_1(x), \hat{\mu}_0(x)$$ as in T-learner <br> 2. Impute: $$D_1 = Y - \hat{\mu}_0(x), \quad D_0 = \hat{\mu}_1(x) - Y$$ <br> 3. Fit $$g_1(x), g_0(x)$$ to imputed ITEs <br> 4. Estimate $$e(x) = P(T=1 \mid X=x)$$ | $$\hat{\tau}(x) = e(x) \cdot g_0(x) + (1 - e(x)) \cdot g_1(x)$$                                                    | Handles imbalance well; leverages both groups                                                 | Requires multiple models; sensitive to propensity or outcome model misspecification                        |
| F-learner   | $$\mathbb{E}[\tilde{Y} \mid X = x]$$, with transformed outcome $$\tilde{Y}$$ | Model $$\hat{g}(x) \approx \mathbb{E}[\tilde{Y} \mid X = x]$$, where $$\tilde{Y} = \frac{Y - \bar{Y}}{p(1-p)}(T - p), \quad p = P(T=1)$$                                    | $$\hat{\tau}(x) = \hat{g}(x)$$                                                                                      | Works with any ML model on transformed $$\tilde{Y}$$                                         | High variance; works better with large datasets                                                             |
| R-learner   | Orthogonalized residual treatment effect         | 1. Estimate: $$\hat{m}(x) = \mathbb{E}[Y \mid X=x]$$, $$\hat{e}(x) = P(T=1 \mid X=x)$$ <br> 2. Minimize: $$\sum_i ( (Y_i - \hat{m}(X_i)) - (T_i - \hat{e}(X_i)) g(X_i) )^2$$                                     | $$\hat{\tau}(x) = \hat{g}(x)$$                                                                                      | Reduces bias from nuisance parameters; integrates well with orthogonalization and DML         | Requires reliable nuisance models; more computationally intensive                                           |

## Causal Forests



## Challenges

Estimating heterogeneous treatment effects (CATE) is essential for personalization and targeting, but it is substantially harder and noisier than estimating overall average treatment effects (ATE). The fundamental difficulty is that we only observe one outcome for each individual. This means CATE is estimated in expectation over groups of individuals with similar covariates, not at the individual level.

Several challenges follow:

- Small sample sizes within groups: As we condition on more features (high-dimensional covariates), the number of individuals sharing the same covariates shrinks, making estimates noisier.
- Uneven treatment assignment: If some subgroups receive more or less treatment, naive comparisons can mistake differences due to treatment range or assignment imbalance for real effect heterogeneity.
- Non-linear relationships: Effect heterogeneity may not come from true differences in how groups respond, but instead from non-linear transformations of either the treatment or the outcome. This is especially problematic with binary outcomes, but can also affect continuous treatments.

### Challenges Specific to Binary Outcomes

Binary outcomes (like conversion) introduce additional difficulties:

- Non-linearity of probabilities: We are estimating probabilities, which are typically modeled with nonlinear link functions (such as logistic or probit). As a result, even constant latent treatment effects translate to nonlinear observed differences.
- Baseline-dependent marginal effects: The effect of a treatment depends on the starting probability. For example, increasing a probability from $$0.05$$ to $$0.15$$ may seem like a large relative effect, but increasing from $$0.8$$ to $$0.9$$ may be harder to achieve in practice, despite being numerically smaller.
- Response curve shape: These models often follow an S-shaped (sigmoid) response curve. Treatment effects tend to be largest around the midpoint (approximately $$50\%$$) and smallest at the extremes (near $$0\%$$ or $$100\%$$). This means that even variables unrelated to the true treatment effect (such as baseline motivation) can produce apparent heterogeneity purely due to where they sit on this curve.
- High variance in observed effects: Binary outcomes inherently introduce noise. Even identical individuals may have different outcomes purely by chance. This variance is particularly problematic with rare events, where small random fluctuations lead to exaggerated apparent effects.

### Focusing on Decisions, Not Just Estimates

Fortunately, we usually do not need precise individual-level CATE estimates. In practice, we want good decisions about who to prioritize. Several strategies help shift focus from noisy point estimates to decision-relevant metrics:

- Uplift curves (or gain charts): Plot the cumulative benefit as you target increasing proportions of the population, ranked by predicted uplift. Useful for choosing how much of your population to target.
- Ranking scores: Often, ordering individuals by expected benefit matters more than the exact magnitude of their estimated CATE. For example, targeting the top $$10\%$$ most responsive customers.
- Binned or aggregated estimates: By grouping individuals into bins (for example, deciles of predicted uplift), you can reduce variance and provide more stable, interpretable summaries of treatment effects. This is often closer to how targeting decisions are actually made at scale.

By emphasizing stable rankings and group-level effects over noisy individual estimates, you can still make effective personalized decisions even when CATE estimates are imperfect.


---
# Other common methods

## Mediation analysis

Mediation analysis is a statistical technique used to understand the mechanism or causal pathway by which a treatment (or exposure) influences an outcome through an intermediate variable known as the mediator. Rather than simply estimating whether a treatment affects an outcome, mediation analysis seeks to quantify $$\text{how much}$$ of this effect occurs $$\text{directly}$$ and $$\text{how much}$$ is transmitted $$\text{indirectly}$$ through the mediator.

### Key concepts

- Treatment (or exposure), $$T$$: The variable or intervention whose effect on the outcome we want to study.  
- Mediator, $$M$$: An intermediate variable that lies on the causal path between treatment and outcome.  
- Outcome, $$Y$$: The final variable of interest that we want to explain or predict.

### Goals

The primary goal of mediation analysis is to decompose the total effect of the treatment on the outcome into two components:

- Direct effect: The portion of the treatment effect on the outcome that is not transmitted through the mediator. This captures pathways where treatment influences the outcome independently of the mediator.  
- Indirect effect: The portion of the treatment effect that operates through the mediator. This represents the causal pathway where treatment changes the mediator, which in turn affects the outcome.

Mathematically, this decomposition helps to clarify how the treatment brings about change in the outcome, offering richer insights beyond the average treatment effect (ATE).

### Typical workflow

Mediation analysis is generally conducted on individual-level data collected in a single study or experiment. The most common approach involves a series of regression models to estimate direct and indirect effects:

1. Estimate the total effect of $$T$$ on $$Y$$:  
   Fit a regression model predicting $$Y$$ from $$T$$ (and relevant covariates). The coefficient of $$T$$ here captures the total effect.

2. Estimate the effect of $$T$$ on the mediator $$M$$:  
   Fit a regression model predicting $$M$$ from $$T$$ (and covariates). This captures how the treatment influences the mediator.

3. Estimate the effect of the mediator $$M$$ on the outcome $$Y$$, controlling for $$T$$:  
   Fit a regression model predicting $$Y$$ from both $$T$$ and $$M$$ (and covariates). The coefficient on $$M$$ here captures the mediator’s effect on the outcome, while the coefficient on $$T$$ estimates the direct effect.

4. Calculate the indirect effect:  
   Multiply the effect of $$T \to M$$ by the effect of $$M \to Y | T$$ to estimate the indirect effect.

5. Calculate the direct effect:  
   The coefficient of $$T$$ in the outcome model controlling for $$M$$ is the direct effect.

6. Inference and uncertainty:  
   Since the indirect effect is a product of two coefficients, its sampling distribution is usually non-normal. Bootstrapping or other resampling techniques are commonly used to estimate confidence intervals and test significance.

### Common methods

- Regression-based mediation: The classical Baron and Kenny approach uses linear regression models to estimate paths.  
- Structural Equation Modeling (SEM): A flexible framework that models direct and indirect pathways simultaneously, allowing for complex mediators and latent variables.  
- Causal Mediation Analysis: Uses counterfactual frameworks to define natural direct and indirect effects, allowing for more precise causal interpretation under stronger assumptions.

### Assumptions

For mediation analysis to yield valid causal interpretations, several assumptions must hold:

- No unmeasured confounding between treatment and outcome, treatment and mediator, and mediator and outcome (conditional on treatment and covariates).  
- Correct model specification of the relationships between variables.  
- No interference between units (treatment or mediator for one individual does not affect another).  
- Temporal ordering: The mediator occurs after the treatment and before the outcome.

By carefully applying mediation analysis, researchers can gain insights not only about whether a treatment works, but also how it works, illuminating the causal mechanisms underlying observed effects.

### Meta mediation analysis

Meta-mediation extends this idea to a higher level. Instead of analyzing individual-level data, meta-mediation uses summary effect estimates from multiple independent experiments or studies (meta-experiments). It estimates how much of the variation in treatment effects on the outcome across experiments can be explained by variation in treatment effects on a mediator. In other words, meta-mediation performs mediation analysis across experiments, using the estimated treatment effects on the mediator and outcome from each experiment to infer causal mediation pathways at the meta-experiment level.

Suppose you have multiple experiments (meta-experiments), each perturbing some feature and measuring:

- An offline metric $$ M $$ (e.g., search relevance score),
- An online metric $$ O $$ (e.g., click-through rate),
- A final business KPI $$ Y $$ (e.g., revenue).

Our goal is to understand how much the offline metric $$ M $$ affects the business KPI $$ Y $$, and how much of this effect is mediated through the online metric $$ O $$.

Imagine $$ n $$ independent experiments, each producing estimated treatment effects on $$ M $$, $$ O $$, and $$ Y $$:

- $$ \hat{\delta}_M^{(i)} $$ = treatment effect on offline metric,
- $$ \hat{\delta}_O^{(i)} $$ = treatment effect on online metric,
- $$ \hat{\delta}_Y^{(i)} $$ = treatment effect on business KPI,

where $$ i = 1, 2, \ldots, n $$.

We assume the causal relations:

$$
\hat{\delta}_O^{(i)} = \beta_{MO} \hat{\delta}_M^{(i)} + \epsilon_O^{(i)}
$$

$$
\hat{\delta}_Y^{(i)} = \beta_{OY} \hat{\delta}_O^{(i)} + \epsilon_Y^{(i)}
$$

Here,

- $$ \beta_{MO} $$ is the causal effect of the offline metric on the online metric,
- $$ \beta_{OY} $$ is the effect of the online metric on the business KPI,
- $$ \epsilon_O^{(i)} $$, $$ \epsilon_Y^{(i)} $$ are noise terms.

### Two-Stage Regression

We perform two regressions across the meta-experiments:

1. Regress $$ \hat{\delta}_O $$ on $$ \hat{\delta}_M $$:

$$
\hat{\delta}_O^{(i)} = \alpha_1 + \hat{\beta}_{MO} \hat{\delta}_M^{(i)} + u_i
$$

2. Regress $$ \hat{\delta}_Y $$ on $$ \hat{\delta}_O $$:

$$
\hat{\delta}_Y^{(i)} = \alpha_2 + \hat{\beta}_{OY} \hat{\delta}_O^{(i)} + v_i
$$

The total mediated effect of the offline metric on the business KPI is estimated by the product:

$$
\hat{\beta}_{\text{total}} = \hat{\beta}_{MO} \times \hat{\beta}_{OY}
$$

### Estimating Confidence Intervals (CIs) Using Bootstrap

Because the total effect is a product of two regression coefficients, its uncertainty is non-trivial. To estimate CIs:

- Resample the meta-experiments with replacement many times (e.g., 1000 bootstraps).
- For each bootstrap sample:
  - Recompute $$ \hat{\beta}_{MO} $$ and $$ \hat{\beta}_{OY} $$,
  - Compute the product $$ \hat{\beta}_{MO} \times \hat{\beta}_{OY} $$.
- Use the empirical distribution of these products to derive confidence intervals, typically by taking the 2.5th and 97.5th percentiles for a 95% CI:

$$
\text{CI}_{95\%} = \left[ \text{percentile}_{2.5\%}(\hat{\beta}_{\text{total}}), \quad \text{percentile}_{97.5\%}(\hat{\beta}_{\text{total}}) \right]
$$

### Interpretation

- $$ \hat{\beta}_{MO} $$ quantifies how changes in the offline metric affect the online metric.
- $$ \hat{\beta}_{OY} $$ quantifies how changes in the online metric affect the business KPI.
- Their product $$ \hat{\beta}_{\text{total}} $$ captures how the offline metric ultimately influences the business KPI through the online metric.
- Bootstrapped confidence intervals reflect uncertainty from both regression steps and their interaction.

### Assumptions

This meta-mediation approach relies on several key assumptions to produce valid causal estimates:

1. **Randomization and No Unmeasured Confounding**  
   Each experiment in the meta-analysis is assumed to be a randomized controlled trial or equivalent, ensuring that treatment assignment is independent of potential outcomes. This justifies interpreting estimated effects $$ \hat{\delta}_M^{(i)}, \hat{\delta}_O^{(i)}, \hat{\delta}_Y^{(i)} $$ as unbiased causal effects.

2. **Linearity and Additivity of Effects**  
   The model assumes a linear relationship between the offline metric and the online metric, and between the online metric and the business KPI:

   $$
   \hat{\delta}_O^{(i)} = \beta_{MO} \hat{\delta}_M^{(i)} + \epsilon_O^{(i)}
   $$

   $$
   \hat{\delta}_Y^{(i)} = \beta_{OY} \hat{\delta}_O^{(i)} + \epsilon_Y^{(i)}
   $$

   This means the mediation is additive and effects do not vary nonlinearly or interact with other unmodeled factors.

3. **No Direct Effect of Offline Metric on Business KPI Outside the Mediator**  
   The framework assumes that all causal influence of the offline metric $$ M $$ on the business KPI $$ Y $$ operates through the online metric $$ O $$ (the mediator). There is no unmodeled direct pathway from $$ M $$ to $$ Y $$ that bypasses $$ O $$.

4. **Independent and Identically Distributed (i.i.d.) Errors**  
   The noise terms $$ \epsilon_O^{(i)} $$ and $$ \epsilon_Y^{(i)} $$ are assumed to be independent across experiments and have mean zero. This justifies the use of standard regression techniques and bootstrap inference.

5. **Sufficient Variation Across Experiments**  
   The meta-experiments must provide enough variation in the treatment effects on $$ M $$ and $$ O $$ to reliably estimate the regression coefficients $$ \beta_{MO} $$ and $$ \beta_{OY} $$.

6. **Stable Unit Treatment Value Assumption (SUTVA)**  
   There are no interference effects between units or experiments — that is, the treatment effect for one experiment is unaffected by others.

Violations of these assumptions can bias estimates or invalidate confidence intervals. For example, if the offline metric affects the business KPI through pathways other than the online metric, the mediation estimate will be incomplete. Similarly, unmeasured confounding or nonlinear effects require more advanced methods beyond this linear meta-mediation framework.

### Summary

Meta-mediation can estimate how a treatment’s effect on an upstream (offline) metric transmits through a mediator (online metric) to impact a final outcome (business KPI). It focuses on decomposing causal pathways using aggregated effects from multiple experiments.

This methodology helps organizations understand which offline improvements lead to better business outcomes by driving changes in online metrics, guiding metric prioritization and investment. In practice, you often use propensity scores or matching to reduce confounding before performing mediation analysis, or you use IV methods to address confounding in mediator or treatment assignment when needed.

## Front-Door Adjustment

---
# Causal Methods Summary

| Category                         | Method                           | Focus          | Summary                                                                                                   | Key Assumptions / Applicability                              | Key Strengths                                        | Limitations                                           | Typical Use Cases                             |
|----------------------------------|----------------------------------|----------------|-----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| **Randomized Experiments**       | Randomized Controlled Trial (RCT) | ATE            | Gold standard using random treatment assignment.                                                          | Randomization removes confounding                             | High internal validity                              | Expensive or infeasible in some settings              | Clinical trials, A/B testing                          |

| **Quasi-Experimental Designs**   | Difference-in-Differences (DiD)   | ATE            | Compares before-after changes in treated vs control groups.                                               | Parallel trends assumption                                    | Controls for time-invariant confounding             | Requires strong parallel trend assumption             | Policy evaluation                                     |
|                                  | Regression Discontinuity (RDD)    | Local ATE       | Uses cutoff-based assignment to estimate local causal effects.                                            | Continuity of potential outcomes at threshold                 | Strong internal validity locally                   | Only estimates local effect, requires no sorting       | Eligibility thresholds                                 |
|                                  | Instrumental Variables (IV)       | Local ATE       | Uses instruments correlated with treatment but not directly with outcome.                                 | Valid instrument (exclusion restriction, relevance)           | Addresses unobserved confounding                   | Requires strong instrument validity                    | Economics, natural experiments                         |
|                                  | Synthetic Control                 | ATE            | Constructs a weighted combination of control units as a counterfactual.                                   | Availability of suitable controls                             | Good for case studies                              | Only one treated unit, sensitive to specification      | Comparative case studies                              |

| **Adjustment-Based (Observational)** | Regression Adjustment         | ATE / CATE     | Regress outcome on treatment and covariates.                                                              | Model correctly specified                                     | Simple and interpretable                           | Sensitive to functional form assumptions              | Broad observational applications                      |
|                                  | Propensity Score Matching (PSM)  | ATE / CATE     | Matches treated and control units with similar covariates.                                                | No unmeasured confounding                                     | Intuitive, common practice                          | Discards unmatched units, poor high-dimensional match  | Healthcare, social sciences                           |
|                                  | Inverse Probability Weighting (IPW) | ATE / CATE   | Weights samples by inverse probability of treatment.                                                      | Correct propensity model                                      | Uses all data, reduces imbalance                    | Can be unstable with extreme weights                   | Longitudinal studies                                  |
|                                  | Targeted Maximum Likelihood Estimation (TMLE) | ATE / CATE | Semi-parametric method using ML for double robustness.                                                    | No unmeasured confounding                                     | Efficient, integrates flexible models               | Complex implementation                                | Biostatistics, epidemiology                           |
|                                  | Orthogonal / Double ML           | ATE / CATE     | Residualizes treatment and outcome before second-stage estimation.                                        | All confounders measured; proper cross-fitting                | Reduces bias from flexible models                   | Sensitive to nuisance model errors                     | High-dimensional data, ML integration                 |
|                                  | Generalized Propensity Score     | Continuous Treatment | Extension of propensity scores for dose-response estimation.                                        | No unmeasured confounding                                     | Works for non-binary treatments                     | Complex estimation and overlap issues                  | Dose-response studies                                 |

| **Heterogeneity-Focused (CATE)** | Meta-Learners (S-, T-, X-, F-)   | CATE           | Wrap standard ML models to estimate CATE using different strategies.                                      | No unmeasured confounding                                     | Flexible, modular                                    | Can overfit or be unstable in small samples           | Personalization, marketing                            |
|                                  | R-Learner                        | CATE           | Residualizes both outcome and treatment before modeling effect.                                           | Similar to orthogonal ML                                     | Plug-in regression model, orthogonalizes confounding | Requires accurate nuisance models                     | General CATE estimation                               |
|                                  | Causal Forests / GRF             | CATE           | Tree-based ensemble splits to maximize treatment effect heterogeneity.                                   | Honest splitting, sufficient overlap                         | Captures non-linear, non-additive effects            | Less interpretable, slower than single trees           | Subgroup discovery, uplift modeling                   |
|                                  | BART (Bayesian Additive Trees)   | CATE / ATE     | Bayesian ensemble of trees that models uncertainty in predictions.                                        | Prior specification, ignorable treatment assignment          | Non-parametric, uncertainty estimates                | Computationally expensive                             | Economics, health outcomes                            |
|                                  | TARNet                           | CATE           | Deep network learns shared representation then splits into treated/control outcome heads.                 | No unmeasured confounding                                     | Flexible, treatment-agnostic architecture            | Requires large data                                    | Individual effect prediction                          |
|                                  | CFRNet                           | CATE           | TARNet + IPM regularization to encourage balanced representations.                                        | Balanced representation learning                              | Improves generalization to counterfactuals           | Needs careful tuning of loss terms                    | Counterfactual prediction                             |
|                                  | DragonNet                        | CATE           | Extends TARNet with a joint propensity prediction head and targeted loss.                                 | Propensity model and outcome learning combined                | Improves efficiency and bias correction              | More complex training setup                           | Personalized interventions                            |
|                                  | GANITE                           | CATE / ITE     | Uses adversarial learning to generate counterfactual outcomes.                                            | Requires valid GAN setup, large data                         | Models counterfactuals directly                      | Less interpretable, hard to train                     | Research on ITE, simulations                          |

| **Mechanism / Pathway**          | Mediation Analysis               | Decomposition   | Decomposes treatment effect into direct and indirect via mediator.                                        | Sequential ignorability                                     | Clarifies causal pathways                           | Sensitive to mediator-outcome confounding             | Psychology, healthcare                                |
|                                  | Front-Door Adjustment            | ATE            | Identifies effect using mediator when back-door paths cannot be blocked.                                  | Correct causal graph, measured mediator                      | Identifies effects with unmeasured confounders       | Strict assumptions, hard to implement                 | Rare, but useful when applicable                      |

| **Robustness & Sensitivity**     | Sensitivity Analysis             | Other           | Quantifies how robust conclusions are to unmeasured confounding.                                          | Varies by method                                             | Helps stress-test assumptions                        | Does not identify causal effect alone                 | Complementary diagnostic                              |
|                                  | Negative Controls                | Other           | Uses variables known to have no effect to detect bias or confounding.                                     | Valid negative controls                                     | Useful for detecting residual confounding            | Needs careful selection of controls                   | Epidemiology, observational studies                   |




---

## What is the story?

Is there a clear story



---
## Application in industry

Working in industry, with a pressure to move fast, the careful consideration part is often lacking. Therefore, I generally think that even if it's just a white board exercise, it can be extremely helpful to attempt a quick causal graph of your problem. Bring in both product and technical stakeholders to help you assess your assumptions, and don't forget to save it (e.g. take a picture). It can also be helpful to note down additional information like whether a variable is expected to have a positive or negative effect on the outcome (a signed graph). Note, a common response you might get to this exercise is something along the lines of "shouldn't we let the data tell us, rather than assuming". While this is tempting, it's often impossible. For example, let's say you see an association between A and B, this doesn't tell us which causal relationship is true (e.g. does A cause B; B cause A; or are A and B both caused by an unmeasured confounder). However, in some cases, causal discovery from data is possible. For example, if we know the time sequencing we may be able to deduce that A occurs first and then B, limiting the possible relationships.


Add examples from different companies that have publicly shared

In my experience, it's come down to whether there's expertise in the team to understand causality. If senior DS support it, and leadership trusts them, then it will be taken seriously. However, you need some level of critical mass. It's a very hard job convincing people if it's not the norm and people are ignorant of the challenges.

It's particularly hard given so much is untestable. You always get a result, and it can look plausible, but it will always be biased because of unmeasured confounders or mis-specification of the problem. As a DS you're trying to convince stakeholders to interpret results within the bounds justified, and often they will not.

People are often more use to using ML, with the purpose of predicting some outcome, and objective measures of evaluation like mean squared error. Even if it's a black box model, we can assess the performance against our goal of prediction. In contrast, causal inference requires careful consideration of what is included, and whether a method is appropriate. 

In causal inference everything is more difficult, at least IMO. You can't use a variable without careful consideration, not all methodologies are appropriate, and you don't know how close your treatment effect estimate is to the "true" value.

---
References:
-  Hern´an MA, Robins JM (2020). [Causal Inference:
What If](https://static1.squarespace.com/static/675db8b0dd37046447128f5f/t/677676888e31cc50c2c33877/1735816881944/hernanrobins_WhatIf_2jan25.pdf). Boca Raton: Chapman & Hall/CRC.
- Inoue, K., Goto, A., Sugiyama, T., Ramlau-Hansen, C. H., & Liew, Z. (2020). The Confounder-Mediator Dilemma: Should We Control for Obesity to Estimate the Effect of Perfluoroalkyl Substances on Health Outcomes? Toxics, 8(4), 125. https://doi.org/10.3390/toxics8040125
- Pearl
- Rubin
- https://booking.ai/encouraged-to-comply-3cd95b447a20 

## Other issues

-- add network effects?

Even in a randomised trial, there are still some potential issues, for example interference and leakage. Interference is when an individual’s outcome is influenced by their interaction with other population members. In this scenario the counterfactual is not well defined because the outcome depends on the other individual's treatment values. For example, the causal effect of person A starting tennis on some health marker, may depend on whether their friend's are also assigned to play tennis. You can try to restrict the question to remove the interference, but this can become difficult to measure and doesn't necessarily answer the question of interest e.g. the causal effect of person A starting tennis when no friends play. A similar but different issue is leakage, where individuals receive the incorrect treatment (e.g. I send you an offer and you share it with a friend who is supposed to be in the control group). This typically dilutes the effect, but the counterfactual can still be defined.

| **Aspect**                   | **Interference**                                             | **Leakage of Treatments**                                    |
|------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| **Definition**                | When an individual's outcome is influenced by the treatment or behavior of others. | When individuals in the control group (or other non-treatment groups) receive some form of treatment. |
| **Context**                   | Common in social networks, educational programs, or competitive environments where interactions occur. | Common in experimental trials, especially when participants don’t follow treatment protocols. |
| **Impact on Counterfactuals** | The counterfactual is not well-defined because an individual’s outcome depends on others' treatments. | The counterfactual can still be defined, but it becomes biased due to treatment contamination. |
| **Cause**                     | Results from **interactions between individuals** (e.g., peer effects, social contagion). | Results from **imperfect treatment assignment** or **adherence** to treatment protocols (e.g. non-compliance could be considered a type of leakage). |
| **Example**                   | In an educational study, one student’s performance may depend on whether their peers received the treatment. | In a clinical trial, control group participants accidentally receiving the treatment drug. |
| **Effect on Treatment Effect** | Makes it difficult to estimate treatment effects due to non-independence of outcomes. | Dilutes the treatment effect, making it harder to assess the true effectiveness of the treatment. |
| **Type of Problem**           | A **structural problem** in defining counterfactuals due to interdependence between individuals. | A **practical problem** with treatment implementation, leading to contamination between groups. |



## Random variability

Up to now we've assumed we're working with the full population, and there's no variability in the estimates. However, we in fact have two sources of random error: 1) sampling variability; and, 2) nondeterministic counterfactuals.

1. **Sampling Variability**: I've talked about this in almost all the posts, but we typically can't get data on the whole population. Instead we take a random sample and use that to estimate the population parameters. Of course the estimates will not exactly equal the population value, but assuming it is random error, the larger the sample the smaller the gap should be (law of large numbers). We will address this later, describing statistical tests.

2. **Nondeterministic counterfactuals**: The main idea behind nondeterministic counterfactuals is that we do not assume a fixed, deterministic outcome for each individual under each treatment condition. Instead, we recognize that the outcome for an individual is random and follows a certain probability distribution. This means that even if individuals receive the same treatment, they might experience different outcomes due to inherent variability. 

For example, let’s consider an ad campaign:
- If the customer sees the ad (t=1), the probability they will purchase is 0.5 (50% chance of purchasing). The expected outcome under treatment (seeing the ad) is $$E[Y_1]=0.5$$ (the average probability of purchase).
- If the customer does not see the ad (t=0), the probability of purchase might be 0.2 (a 20% chance of purchasing) and $$E[Y_0]=0.2$$.

For any individual, the actual outcome is random and depends on the treatment, but we can calculate expected outcomes by integrating over the possible outcomes using their respective probabilities.



Idea: Use instrumental variables to eliminate the measurement error — specifically, use a leave-one-out version of the proxy metric to remove its noise from the equation.

Steps:

For each experiment, create a version of the proxy metric’s treatment effect that does not include that experiment’s data (like jackknifing).

Use this “instrumented” version to predict the effect on the north star.

This gets around the noise correlation because:

The instrument (the proxy excluding the experiment’s own data) is uncorrelated with that experiment’s noise.

📌 No need to assume homogeneous error structure, so it’s more flexible than TC.

📈 Good for: Heterogeneous experiments, where noise varies between experiments.










Common conditional mean models:
- Generalised Linear Model (GLM) which have 2 components: a linear functional form; and a link function which constrains the output values (e.g. log for y>0; logit for y between 0 and 1; identity for continuous y);
- Generalized Estimating Equations (GEE): a method for estimating population-averaged (marginal) effects in correlated or clustered data (e.g. multiple observations over time or within a cluster);
- Generalized additive model (GAM) is an extension of the generalized linear model (GLM) that relaxes the assumption of linearity in the covariates. Rather than assuming a linear relationship between predictors and the response, GAMs allow for smooth, potentially nonlinear effects of each covariate.
- Kernel regression: Kernel regression is a nonparametric technique that estimates the conditional expectation $$E[Y \mid X=x]$$ by averaging nearby observations of Y, with weights decreasing with distance from the target point x. It does not assume that the relationship between X and Y is linear, quadratic, or of any specific form — making it fully data-driven.



### Example

Imagine you work on the Apple Watch’s breathing app and want to understand whether it lowers users’ resting heart rate (RHR) over a two-week period. Some users engage with the app frequently, while others don’t use it at all—but these groups likely differ in important ways. For instance, breathing app users may be more health-conscious, younger, and exercise more, but also more stressed (which might drive them to use the app). These factors can also affect RHR, making them potential confounders. 

To make a fair comparison, we use matching to balance these observed characteristics across frequent and infrequent users. We match on pre-treatment variables that proxy for confounders: baseline RHR, heart rate variability (as a stress indicator), hours of sleep, daily steps, and weekly workouts (as measures of fitness). We also control for the time of day RHR is measured to reduce noise in the outcome. However, we avoid adjusting for post-treatment variables like sleep, fitness, or app engagement, since these could be colliders—affected by both app usage and RHR. We also avoid controlling for mediators like post-treatment stress, which lie on the causal path from treatment to outcome. 
