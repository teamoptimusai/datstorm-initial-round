# DataStorm Initial Round
![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)
### Data Analysis
By checking the info of the training dataset, We can see that there are 6 Categorical Data and No columns with null data.
![correlation_maps](https://i.ibb.co/r7W7tYx/Capture.png)

**'​ PAY_OCT', 'DUE_AMT_SEP', 'DUE_AMT_OCT', 'DUE_AMT_NOV', 'DUE_AMT_DEC','DUE_AMT_JULY'** ​ are highly correlated in **​ train dataset ​ and ​ 'PAY_OCT', 'DUE_AMT_SEP','DUE_AMT_OCT', 'DUE_AMT_NOV', 'DUE_AMT_JULY' ​** are highly correlated in the ​ test dataset.

![correlation_maps](https://i.ibb.co/rtQ02n5/Capture2.png)

None of the life factors ('Gender', 'Age', 'Marriage_status', 'Education_status') are correlated. As you can see most of the due amounts are highly correlated. Unlike other due amounts Due amount of December shows significant importance in the Gender factor.

**In Conclusion,** the ​ Age ​ and ​ Education Status ​ plays a huge role in predicting whether the client will default or not in the next month. Furthermore, due amounts have a higher correlation except in the month of December. So, a huge concern should go to the values in December due amounts. So, it is ​ better to take a sum of the due amounts as another feature ​, so we can remove the other due amounts. There is very little correlation in paid amounts of each month, so there is no need to fiddle with them.

### Data Cleaning
Since the column "Balance_limit_V1' is in a form that we cannot use directly on the classifier we had to convert it manually.
```python
data_comp[​'Balance_Limit_V1'​] = data_comp[​'Balance_Limit_V1'​].map({​'100K'​:100000 ​, ​'200K'​: ​ 200000 ​, ​'300K'​: ​ 300000 ​,'400K'​: ​ 400000 ​, ​' 500K'​: ​ 500000 ​, ​'1M'​: ​ 1000000 ​, ​'1.5M'​: ​ 1500000 ​, ​'2.5M'​:2500000 ​})
```
Then we applied "One Hot Encoding" to the categorical data on the data frame
```python
data_comp = pd.get_dummies(data_comp,columns=[​'Gender'​, ​'EDUCATION_STATUS'​,'MARITAL_STATUS'​, ​'AGE'​])
```
Then used normalizer to standardize the data. Hence it would perform efficiently in classifiers
```python
from​ sklearn.preprocessing ​import​ MinMaxScaler, StandardScaler,Normalizer
preprocessing.scale(data_comp[[​'Balance_Limit_V1'​, ​'PAY_JULY'​, ​'PAY_AUG'​,'PAY_SEP'​, ​'PAY_OCT'​,​'PAY_NOV'​,​'PAY_DEC'​, ​'DUE_AMT_JULY'​, ​'DUE_AMT_AUG'​,'DUE_AMT_SEP'​, ​'DUE_AMT_OCT'​, ​'DUE_AMT_NOV'​, ​'DUE_AMT_DEC'​,'PAID_AMT_JULY'​, ​'PAID_AMT_AUG'​, ​'PAID_AMT_SEP'​, ​'PAID_AMT_OCT'​,'PAID_AMT_NOV'​, ​'PAID_AMT_DEC'​]])
```
### Feature Engineering
We used several methods to enhance the given features to get a better fit for the Classification models. Using the Results we got from our Data Analysis, we made 3 new features [TOT_DUE], [TOT_PAY] and [BAL]
```python
data_comp[​'TOT_DUE'​] = data_comp[​'DUE_AMT_JULY'​] + data_comp[​'DUE_AMT_AUG'​] + data_comp[​'DUE_AMT_SEP'​] + data_comp[​'DUE_AMT_OCT'​] + data_comp[​'DUE_AMT_NOV'​] + data_comp[​'DUE_AMT_DEC'​]

data_comp[​'TOT_PAY'​] = data_comp[​'PAID_AMT_JULY'​] + data_comp[​'PAID_AMT_AUG'​] + data_comp[​'PAID_AMT_SEP'​] + data_comp[​'PAID_AMT_OCT'​] + data_comp[​'PAID_AMT_NOV'​] + data_comp[​'PAID_AMT_DEC'​]

data_comp[​'BAL'​] = data_comp[​'TOT_DUE'​] - data_comp[​'TOT_PAY'​]
```
### Classification
First we used several classifiers on the modified dataset namely Logistic Regression, Decision Tree Classification, Gradient Boost, Ada Boost, Random Forest, Extra Tree, K-Neighbors, Support Vector Classifications and Gaussian Naive Bayes. Following are the Optimum Results we got from those Classifiers.
```python
Logistic Regression : ​0.795625
Decision Tree Classification : ​0.721875
Gradient Boosting Classification : ​0.80875
Ada Boosting Classification : ​0.8064583333333334
RandomForest Classification : ​0.8064583333333334
Extra Tree Classification : ​0.7991666666666667
K-Neighbors Classification : ​0.7777083333333333
Support Vector Classification : ​0.8083333333333333
Gaussian Naive Bayes : ​0.6914583333333333
```
Since we didn't get the expected accuracy, we dropped highly correlated (>0.85) features. The
Following are them.
```python
[​'DUE_AMT_SEP'​, ​'DUE_AMT_OCT'​, ​'DUE_AMT_JULY'​, ​'DUE_AMT_NOV'​, ​'Gender_F'​,'MARITAL_STATUS_Other'​]
```
Then we used a LGBM Classifier because it has a higher number of tunable parameters unlike the previous classification methods. First we manually fiddled with the parameters but it ended up in a disaster. So, we used **​ Bayesian Optimization ​**with the help of our lgbm function to obtain the set of parameters.
```python
params = {​'num_leaves'​: ​ 78 ​, ​'min_data_in_leaf'​: ​ 15 ​, ​'min_split_gain'​:
0.0027913158608962284​, ​'objective'​: ​'binary'​, ​'max_depth'​: ​ 4 ​,
'learning_rate'​: ​0.03340963722563954​, ​'boosting'​: ​'gbdt'​, ​'bagging_freq'​:
5 ​, ​'bagging_fraction'​: ​0.647657770451912​, ​'bagging_seed'​: ​ 11 ​, ​'verbosity'​:
-1​, ​'reg_alpha'​: ​10.76032018246045​, ​'reg_lambda'​: ​12.761403429830967​,
'num_class'​: ​ 1 ​, ​'nthread'​: ​-1​}
```
Using this set of parameters we are able to get the results
```python
                    precision   recall      f1-score     support
                0   0.83        0.95        0.89        3683
                1   0.68        0.35        0.46        1117
accuracy                                    0.81        4800
macro avg           0.75        0.65        0.67        4800
weighted avg        0.79        0.81        0.79        4800
```
### Business Model
When a person signs up for a credit card, the bank can start to collect data regarding the user and the account. Even though most of the cardholders pay their due amounts, significant loss happens because of credit card defaults. This loss is a pure loss since there is no way that this money can be returned to the bank. Therefore, banks would like to find risky accounts and freeze them before they are too late and the damage is too big. But the patterns of these cardholders are not the same and cannot be analyzed using human analysis. That is why data science is used. Because of this model, every account can be checked and analyzed simultaneously, without spending too much money or human force. In addition to that, the model can be improved over time, which is another additional advantage.
