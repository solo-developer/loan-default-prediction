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

| Algorithms     | Accuracy        | F1 Score  | AUC     |
| -------------  |:-------------:  | -----:    | ----:   |
| LightGBM       | 0.8809          | 0.8728    | 0.8809  |
| XGBoost        | 0.8985          | 0.8914    | 0.8985  |
| CatBoost       | 0.9319          | 0.9276    | 0.9319  |
| ExtraTrees     | 0.8261          | 0.8280    | 0.8261  |
| Decision Trees | 0.6882          | 0.6986    | 0.6882  |
| Random Forest  | 0.7354          | 0.7433    | 0.7354  |
| ANN            | 0.7002          | 0.6868    | 0.7727  |
| Neural Network | 0.7061          | 0.7421    | 0.7369  |
| Stacking Ens.  | 0.9468          | 0.9451    | 0.9512  |

## F1-Score Bar Graph Visualization
![Bar graph Visualization](https://github.com/solo-developer/loan-default-prediction/blob/main/images/bargraph-f1.png)

## Significance of Categorical features (Chi-Squared test)
| Variable       | Chi-Squared |
|----------------|-------------|
| EmploymentType | 529.7449    |
| HasCoSigner    | 390.3050    |
| HasDependents  | 306.8506    |
| Education      | 214.0190    |
| MaritalStatus  | 200.3611    |
| HasMortgage    | 133.2520    |
| LoanPurpose    | 127.9342    |

## Significance of Numerical features (Correlation test)
| Feature         | Correlation |
|-----------------|-------------|
| InterestRate    | 0.131273    |
| LoanAmount      | 0.086659    |
| NumCreditLines  | 0.028330    |
| DTIRatio        | 0.019236    |
| LoanTerm        | 0.000545    |
| CreditScore     | -0.034166   |
| MonthsEmployed  | -0.097374   |
| Income          | -0.099119   |
| Age             | -0.167783   |


## Results after feature reduction (LoanPurpose)

| Algorithms     | Accuracy        | F1 Score  | AUC     |
| -------------  |:-------------:  | -----:    | ----:   |
| LightGBM       | 0.8798          | 0.8714    | 0.8798  |
| CatBoost       | 0.9317          | 0.9274    | 0.9317  |
| ExtraTrees     | 0.8279          | 0.8201    | 0.8261  |
| Decision Trees | 0.6881          | 0.6986    | 0.6881  |
| Random Forest  | 0.7345          | 0.7427    | 0.7345  |
| ANN            | 0.6901          | 0.6062    | 0.6918  |

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


