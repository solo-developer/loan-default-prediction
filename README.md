# loan-default-prediction

## The purpose of this project is to compare different algorithms for loan default prediction.

#### Compared algorithms includes
  * CatBoost Algorithm
  * XGBoost Algorithm
  * LightGBM Algorithm
  * ExtraTrees Algorithm
  * Decision Trees Algorithm
  * Random Forest Algorithm
  * ANN
## Results

| Algorithms     | Accuracy        | F1 Score  |
| -------------  |:-------------:  | -----:    |
| Stacking Ens.  | 0.9468          | 0.9451    |
| CatBoost       | 0.9319          | 0.9276    |
| XGBoost        | 0.8985          | 0.8914    |
| LightGBM       | 0.8809          | 0.8728    |
| ExtraTrees     | 0.8261          | 0.8280    |
| Neural Network | 0.7061          | 0.7421    |
| Random Forest  | 0.7354          | 0.7433    |
| Decision Trees | 0.6882          | 0.6986    |
| ANN            | 0.7002          | 0.6868    |


## F1-Score Bar Graph Visualization
![Bar graph Visualization](https://github.com/solo-developer/loan-default-prediction/blob/main/images/bargraph-f1.png)

## ROC-AUC Comparison
![ROC-AUC Visualization](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/ROC-AUC-Comparison.png)

## Feature Importance Visualization
![Feature Importance Visualization](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/feature-importance.png)

## Significance of Categorical features (Chi-Squared test)
![Chi-Squared test Visualization](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/chi-squared.png)

## Matrix Visualization of Categorical features (Cramer's V)
![Matrix Visualization of Categorical features](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/Cramers%20V%20visualuzation.png)

## Significance of Numerical features (Correlation test)
![Correlation test Visualization](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/correlation.png)

## Matrix Visualization of Numerical features (Correlation matrix)
![Correlation matrix Visualization](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/correlation-matrix-with-class-variable.png)
#### Result Interpretation

### Correlation with Default
1. **Age**: Negative correlation with Default (-0.168). As age increases, the likelihood of default decreases.
2. **Income**: Negative correlation with Default (-0.099). Higher income is associated with a lower likelihood of default.
3. **LoanAmount**: Positive correlation with Default (0.087). Higher loan amounts are slightly associated with a higher likelihood of default.
4. **CreditScore**: Negative correlation with Default (-0.034). Higher credit scores are slightly associated with a lower likelihood of default.
5. **MonthsEmployed**: Negative correlation with Default (-0.097). More months employed is associated with a lower likelihood of default.
6. **NumCreditLines**: Positive correlation with Default (0.028). More credit lines are slightly associated with a higher likelihood of default.
7. **InterestRate**: Positive correlation with Default (0.131). Higher interest rates are associated with a higher likelihood of default.
8. **LoanTerm**: Very low positive correlation with Default (0.001). Loan term length has almost no correlation with default.
9. **DTIRatio**: Positive correlation with Default (0.019). Higher debt-to-income ratio is slightly associated with a higher likelihood of default.

### Other Correlations
1. **Age and LoanAmount**: Slight negative correlation (-0.002). Older age slightly correlates with lower loan amounts.
2. **Income and LoanAmount**: Slight negative correlation (-0.001). Higher income is slightly correlated with lower loan amounts.
3. **InterestRate and Income**: Slight negative correlation (-0.002). Higher income is slightly associated with lower interest rates.
4. **LoanAmount and CreditScore**: Very slight positive correlation (0.001). Higher loan amounts are very slightly associated with higher credit scores.

### General Observations
- Age, Income, CreditScore, and MonthsEmployed show a negative correlation with Default, indicating that higher values in these variables generally reduce the likelihood of default.
- LoanAmount, NumCreditLines, InterestRate, LoanTerm, and DTIRatio show a positive correlation with Default, indicating that higher values in these variables generally increase the likelihood of default.
- The correlations between other variables, such as Age and LoanAmount, Income and LoanAmount, InterestRate and Income, and LoanAmount and CreditScore, are generally very slight.


## Results after feature reduction (Stacking Ensemble)

![Results after feature reduction (Stacking Ensemble)](https://github.com/solo-developer/loan-default-prediction/blob/develop/images/Feature%20reduction-stacking%20ensemble.png)

## Results after feature reduction (LoanPurpose,HasMortgage)

| Algorithms     | Accuracy        | F1 Score  | AUC     |
| -------------  |:-------------:  | -----:    | ----:   |
| LightGBM       | 0.8836          | 0.8757    | 0.8836  |
| CatBoost       | 0.9315          | 0.9272    | 0.9315  |
| ExtraTrees     | 0.8235          | 0.8256    | 0.8235  |
| Decision Trees | 0.6882          | 0.6986    | 0.6882  |
| Random Forest  | 0.7243          | 0.7309    | 0.7243  |
| ANN            | 0.6401          | 0.6010    | 0.6373  |

## Results after feature reduction (LoanPurpose,HasMortgage,MaritalStatus)

| Algorithms     | Accuracy        | F1 Score  | AUC     |
| -------------  |:-------------:  | -----:    | ----:   |
| LightGBM       | 0.8852          | 0.8776    | 0.8852  |
| CatBoost       | 0.9313          | 0.9269    | 0.9313  |
| ExtraTrees     | 0.8305          | 0.8325    | 0.8305  |
| Decision Trees | 0.6882          | 0.6986    | 0.6882  |
| Random Forest  | 0.7243          | 0.7397    | 0.7335  |
| ANN            | 0.6115          | 0.5896    | 0.6011  |


