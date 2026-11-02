# Experimental Stage 3: HistGradientBoostingRegressor

## Hypothesis

A small gradient boosting model using lag features and local statistics could capture short-term nonlinear relationships in aggregate CPU utilization and reduce the prediction error of ARIMA(2,0,0) (MAE 0.1926). The configuration was kept fixed; no hyperparameter search was performed, and no holdout observations were used.

## Features and Methodology

To predict the value at time step `t`, only observations preceding `t` are used:

- Lags 1, 2, and 3.
- Trailing moving averages over 2 and 3 observations.
- Local trend: `(y[t-1] - y[t-3]) / 2`.

The training process constructs one-step-ahead samples using these features and `y[t]` as the target. The 24-step forecast is recursive: after each prediction, the predicted value is appended to the historical sequence, and the lag features, moving averages, and trend are recalculated for the next step. The model is fitted independently for each fold.

Fixed configuration:

| Parameter           |           Value |
| ------------------- | --------------: |
| `loss`              | `squared_error` |
| `learning_rate`     |            0.05 |
| `max_iter`          |              50 |
| `max_leaf_nodes`    |               3 |
| `min_samples_leaf`  |               5 |
| `l2_regularization` |             1.0 |
| `early_stopping`    |         `False` |
| `random_state`      |               0 |

The environment used scikit-learn 1.9.1. No dependencies were installed or upgraded.

## Validation

The same 39 expanding-window folds were used: training sets ranging from 48 to 86 observations, a 24-step forecasting horizon, and `step=1`, all within the 110 training observations.

The 936 prediction pairs cover only 62 unique time steps; therefore, the folds and their overlapping errors are not independent observations. No holdout target values were accessed, and `predictions.csv` was neither generated nor modified during this experiment.

## Aggregate Results

MAE, RMSE, and sMAPE were calculated across the 936 prediction pairs. The standard deviation, median, minimum, maximum, and IQR of MAE summarize the distribution of the 39 fold-level MAE values.

| Model                |        MAE |       RMSE |  sMAPE (%) | Fold MAE: Mean |  Std. Dev. |     Median |       Min. |       Max. |        IQR |
| -------------------- | ---------: | ---------: | ---------: | -------------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| ARIMA(2,0,0)         | 0.19255403 | 0.23165528 | 2.03427219 |     0.19255403 | 0.06152409 | 0.16891865 | 0.13181775 | 0.30814823 | 0.06736413 |
| temporal_regression  | 0.19609920 | 0.23495537 | 2.07159790 |     0.19609920 | 0.06090057 | 0.17672146 | 0.12416374 | 0.30999919 | 0.06891293 |
| ARIMA(2,0,1)         | 0.19615851 | 0.23688028 | 2.07168885 |     0.19615851 | 0.06541268 | 0.17370731 | 0.11043414 | 0.30810274 | 0.10574865 |
| ARIMA(1,0,1)         | 0.19631588 | 0.23658917 | 2.07325975 |     0.19631588 | 0.06537837 | 0.17424034 | 0.10999086 | 0.30965608 | 0.10946263 |
| ARIMA(1,0,0)         | 0.19704495 | 0.23473305 | 2.08160897 |     0.19704495 | 0.05899698 | 0.17140545 | 0.13410331 | 0.30955798 | 0.06974793 |
| mean                 | 0.20501318 | 0.24198943 | 2.16562947 |     0.20501318 | 0.05554252 | 0.17633852 | 0.15089017 | 0.30929598 | 0.07343685 |
| HistGradientBoosting | 0.22440068 | 0.28020731 | 2.35920308 |     0.22440068 | 0.10718664 | 0.26307231 | 0.08479851 | 0.38180359 | 0.20800722 |
| exp_smoothing_05     | 0.23441686 | 0.28700729 | 2.46396493 |     0.23441686 | 0.09727917 | 0.25843273 | 0.08115336 | 0.40205560 | 0.18491223 |
| persistence          | 0.23780907 | 0.29362569 | 2.49880930 |     0.23780907 | 0.10578286 | 0.25693415 | 0.08068202 | 0.47373457 | 0.16527375 |
| exp_smoothing_03     | 0.23946510 | 0.28899063 | 2.51805077 |     0.23946510 | 0.09539646 | 0.26487267 | 0.08356236 | 0.38677825 | 0.18805529 |
| moving_average_6     | 0.24806262 | 0.29846738 | 2.60709546 |     0.24806262 | 0.10010869 | 0.29560628 | 0.07933168 | 0.37205710 | 0.18009476 |
| moving_average_12    | 0.26696791 | 0.30968395 | 2.80800100 |     0.26696791 | 0.09193103 | 0.29078410 | 0.11529572 | 0.40010439 | 0.14059771 |
| drift                | 0.27278112 | 0.34166288 | 2.85579814 |     0.27278112 | 0.13306438 | 0.28391577 | 0.08114596 | 0.55903759 | 0.22782377 |
| seasonal_naive_24    | 0.32245725 | 0.36157234 | 3.38432061 |     0.32245725 | 0.02469156 | 0.32008339 | 0.25930823 | 0.35836263 | 0.03541293 |

The results for all eight baseline models are identical to the previously obtained results. The rounded reference results are also reproduced: ARIMA(2,0,0) MAE 0.1926, temporal_regression MAE 0.1961, and mean MAE 0.2050.

HistGradientBoosting increases MAE by 16.54% compared with ARIMA(2,0,0), 14.43% compared with temporal_regression, and 9.46% compared with mean. Its fold-level MAE standard deviation (0.10719) and IQR (0.20801) are also higher than those of these three reference models.

## HistGradientBoosting Metrics by Forecast Horizon

Each forecast step is evaluated across 39 folds. The metrics at each horizon step are also not independent across folds.

| Step |        MAE |       RMSE |  sMAPE (%) |
| ---: | ---------: | ---------: | ---------: |
|    1 | 0.11631382 | 0.15337502 | 1.21497175 |
|    2 | 0.12566351 | 0.16877125 | 1.31370744 |
|    3 | 0.14794456 | 0.19708355 | 1.54817518 |
|    4 | 0.16480572 | 0.22042482 | 1.72441432 |
|    5 | 0.19211953 | 0.24497284 | 2.01174133 |
|    6 | 0.20865237 | 0.26588197 | 2.18517368 |
|    7 | 0.21615862 | 0.27712930 | 2.26339056 |
|    8 | 0.22399562 | 0.29164211 | 2.34543321 |
|    9 | 0.22602801 | 0.28786310 | 2.36916254 |
|   10 | 0.22790979 | 0.29122330 | 2.38945645 |
|   11 | 0.23417859 | 0.29266143 | 2.45549140 |
|   12 | 0.24195072 | 0.29350617 | 2.53938283 |
|   13 | 0.24721257 | 0.30045172 | 2.59626132 |
|   14 | 0.25317182 | 0.30559258 | 2.66015159 |
|   15 | 0.25748876 | 0.30688550 | 2.70832946 |
|   16 | 0.26229345 | 0.30892962 | 2.76043694 |
|   17 | 0.26519464 | 0.30925662 | 2.79319461 |
|   18 | 0.26831180 | 0.31634622 | 2.82778841 |
|   19 | 0.25881289 | 0.30565485 | 2.73051966 |
|   20 | 0.25118861 | 0.30220890 | 2.65053466 |
|   21 | 0.25038370 | 0.30547385 | 2.64416667 |
|   22 | 0.24825584 | 0.29841697 | 2.62401164 |
|   23 | 0.24876312 | 0.29630712 | 2.63162909 |
|   24 | 0.24881822 | 0.29834259 | 2.63334914 |

## Fold-by-Fold Comparison Against ARIMA(2,0,0)

The difference is defined as `MAE_HistGradientBoosting - MAE_ARIMA`; negative values favor HistGradientBoosting.

HistGradientBoosting outperformed ARIMA in 18 folds, tied in 0 folds, and underperformed in 21 folds, using a tie tolerance of `1e-12`. Consecutive overlapping folds do not represent independent wins.

The 39 exact differences are available in [hist_gradient_boosting_arima_fold_deltas.csv](./hist_gradient_boosting_arima_fold_deltas.csv).

## Complete Results Files

- [hist_gradient_boosting_summary.csv](./hist_gradient_boosting_summary.csv): MAE, RMSE, sMAPE, and fold-level dispersion for all evaluated models.
- [hist_gradient_boosting_horizon_metrics.csv](./hist_gradient_boosting_horizon_metrics.csv): Metrics for each forecast horizon step across all evaluated models.
- [hist_gradient_boosting_fold_metrics.csv](./hist_gradient_boosting_fold_metrics.csv): Individual metrics for each of the 39 folds.
- [hist_gradient_boosting_arima_fold_deltas.csv](./hist_gradient_boosting_arima_fold_deltas.csv): Individual fold-level differences compared with ARIMA(2,0,0).

## Conclusion and Limitations

This configuration does not outperform ARIMA(2,0,0), temporal_regression, or mean. Its performance varies considerably across forecast origins: dispersion is high, and several consecutive winning folds may reflect temporal overlap rather than independent evidence.

The evaluation covers a single aggregate time series and only 62 unique time steps. Therefore, it does not support claims of generalization to other time series, forecasting horizons, or datasets.

No further hyperparameter search was conducted, and no additional models were implemented.
