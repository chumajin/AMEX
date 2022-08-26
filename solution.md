Top 16 Solution - Writeup

First of all, we would like to thank kaggle and the staff for hosting such an interesting competition.

# 1. Summary 
 Our solution is based on extensive data cleaning and multi-model weighted average and stacked ensembles using ranked probs. Using extensive data cleaning, our single model was boosted to cv 0.0004~0.0008 compared to the public Raddar's dataset.The final solution (PVT 0.80842/PUB 0.80105) which shaked up to gold zone  is the average of 3 ensemble models:
- LGBM stack with 61 models
- CMA weighted average with 54 models
- CMA weighted average with 71 models

# 2. Extensive data cleaning 
Starting from Raddar cleanned dataset, some other features were cleaned and modified in order to build a second version of the dataset. The idea here was to clean the data even more and add some diversity to the ensemble. 

## 2.1 Example of boosting cv using cleaning data
- LGBM : 0.7976 → 0.7983 ( +0.0007 )
- XGB : 0.7978 → 0.7986 ( + 0.0008 )
- CatBoost :    0.7964 → 0.7968 ( + 0.0004 ) 

 ※ clean data plus minor change 
   - LGBM 5kfold → 15kfold (+0.0013)
   - CatBoost longer earlystop (+0.002 )

# 3. Single model 
## 3.1 Features and modeling

Basically, we used the features and the models of public notebooks.
Thank you for @thedevastator, @ragnar123,@ambrosm,@cdeotte, @roberthatch

## 3.2 Representative best single model in each model type 

| model       | features | kfold | cv     | public lb | private lb | comment                      |
|-------------|----------|-------|--------|-----------|------------|------------------------------|
| LGBM        | 1784     | 15    | 0.7991 | 0.79961   | 0.80679    | dart with early stop         |
| XGB         | 3413     | 5     | 0.7986 | 0.79873   | 0.80662    | pyramid                      |
| CATBoost    | 2290     | 5     | 0.7968 | 0.79832   | 0.80629    |                              |
| MLP         | 446      | 5     | 0.7919 | 0.79102   | 0.79993    | KD using LGBM oof prediction |
| GRU         | 188      | 5     | 0.7948 | 0.79418   | 0.80363    | KD using LGBM oof prediction |
| transformer | 188      | 5     | 0.7932 | 0.79498   | 0.80379    | KD using LGBM oof prediction |

# 4. Ensemble 

We used the two methods for ensemble with ranked probs. One is the LGBM stacking, the other is the CMA (Covariance Matrix Adaptation) Evolution Strategy [REF:https://www.scm.com/doc/params/python/optimizers/cmaes.html] .

And final submission is the average of the following 3 ensemble models(case1～3) ::
| Case   | Ensemble Method | cv       | public lb | private lb | Num of total models | Num of LGBM | Num of XGBoost | Num of CATBoost | Num of 1dcnn | Num of 2dcnn | Num of MLP | Num of TCN | Num of GRU | Num of Transformer |
|--------|-----------------|----------|-----------|------------|------------------------|-------------|----------------|-----------------|--------------|--------------|------------|------------|------------|--------------------|
| Case 1 | LGBM stacking   | 0.801    | 0.80084   | 0.80882    | 61                     | 28          | 11             | 9               | 1            | 1            | 5          | 2          | 3          | 1                  |
| Case 2 | cma             | 0.803159 | 0.80086   | 0.80794    | 54                     | 20          | 12             | 9               | 1            | 1            | 5          | 2          | 3          | 1                  |
| Case 3 | cma             | 0.802794 | 0.80068   | 0.8079     | 55                     | 25          | 14             |     10　         | 0            |   0　         | 3          | 0          | 3          |       0            |


As you can see, the public lb is almost the same, but the private lb shows much better LGBM stacking. We would have been hard pressed to get the gold medal without LGBM stacking.

## 4.1. Final sub (ensemble result of case 1 ~ 3)
The ensemble result of the above case 1-3 is our final submission.
public lb 0.80105, private lb 0.80842 (16th).

※ Note that : LGBM stacking alone (0.80842) has the capability of ranking 6th.

# 5. Relationship of cv and lb 

As for reference, we share the relationship of cv and lb. It is easy to see without looking at the plot with cma, but the lgbm stacking is a clean extension of the cv/lb straight line of the single model (Maybe cma is overfitting…)












# 6. Other tips that worked well 

- knowledge distillation
- dart longer early stop (3000～10000)
- large kfold
- full training
- pseudo labeling


# 7. Didn't work well 
- post process
- using dow average data
- sample weight of LGBM change
- adding focal loss to LGBM


# 8. Late submission result
 After the end of competition we realized that we had 2 ensembles with PVT LB > 0.80920 (3-4th place) - however, were not considered due to lower CV scores.
avg rank of 3 stack models (XGB+LGB+Catboost) - PVT 0.80928, PUB 0.80067, CV 0.8008 
LGB stack with 61 models + meta features (top 50 eng. features) - PVT 0.80926, PUB 0.80072, CV 0.80097

# 9. Team organization 
We used github for code storage, wandb/xxx for experiments tracking and kaggle datasets for OOF storage and sharing with the team.
