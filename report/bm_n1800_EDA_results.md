## Box Polt
I found no visible outliers.

This means:

1. The values range from approximately 2.3 to 10.7.
2. The middle 50% values are possibly between Q1 = 3.1 and Q3 = 8.1.
3. No values exceeded the IQR outlier thresholds.

Dataset cleaned, therefore proved.

## Histogram + KDE (Distribution Plot)

I believe it follows a bimodal distribution. Which means the timings come from two different regimes:
1. Fast regime
2. Slow regime

I think this shows that the benchmark alternates between 2 performance states and not a single steady state. (Warm-up phase vs Steady State phase)

## Time Series Plot
It fluctuates heavily between ~3 and ~10 across all 1800 samples. This means the system repeatedly switches between the “slow” and “fast” states.
There is no obvious warm-up at the beginning only. Instead the system:
1. stabilizes short-term.
2. then jumps to another regime
3. then stabilizes again
4. Repeats

There is multiple regime changes, not just one.

Yulian I think you need to do KMeans or DBSCAN.

## Rolling Mean
Rolling mean oscillates between ~4.4 and ~6.4 throughout the run. If I had a good steady state, I would assume a flat mean after warm up. But in this case it never flattens. So I think there is no single steady state.

Instead:

* mean drops around 750-1000

* rises again around 1000 - 1250

* drops again around 1500 - 1750

This confirms alternating regimes.

## Rolling Variance
Variance is high everywhere. If warm-up only happened 1 time at beginning, we could see variance high at start and low at end. But this is high everywhere. I think (personally), rhe system’s performance is unstable across the entire benchmark.

## First Difference Plot
I recognised huge spikes both positive and negative. The timing jumps sharply both upward and downward. This means there is frequent regime-switching.

## QQ Plot
2 curved clusters clearly split apart. Meaning:
* Abnormal timings
* Never ever normal distribution.
* Maybe I think there are 2 different distributions.

Yulian and Anuj I think you guys need to do change point detections. It is also safe to assume that we can reject anything assuming Gaussian behaviour.

## Skewness & Kurtosis
1. Right-skewed (as right tail longer).
2. Negative kurtosis: distribution is flatter / wider than a normal distribution.