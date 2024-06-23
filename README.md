# loan-default-prediction

## The purpose of this project is to compare different algorithms for loan default prediction.

#### Compared algorithms includes
  * CatBoost Algorithm
  * LightGBM Algorithm
  * ExtraTrees Algorithm
  * Decision Trees Algorithm
  * Random Forest Algorithm
  * ANN

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


## Results

| Algorithms     | Accuracy        | F1 Score  | AUC     |
| -------------  |:-------------:  | -----:    | ----:   |
| LightGBM       | 0.8809          | 0.8728    | 0.8809  |
| CatBoost       | 0.9319          | 0.9276    | 0.9319  |
| ExtraTrees     | 0.8261          | 0.8280    | 0.8261  |
| Decision Trees | 0.6882          | 0.6986    | 0.6882  |
| Random Forest  | 0.7354          | 0.7433    | 0.7354  |
| ANN            | 0.7002          | 0.6868    | 0.7727  |
