# Machine_Learning_Pipeline

## Summary

The repository outlines the process flow and steps for analyzing a data set using machine learning algorithms. The pipeline focuses on binary classification. The following is a summary of the steps:

- Step 1: Loading Data  
Reading a csv into a dataframe

- Step 2: Outlier Removal   
There are various techniques can be used for outlier removal like using standard deviation or percentiles. In the code data lying beyond a certain threshold of standard deviations is being treated as an outlier.

-  Step 3: Descriptive Statistics     
Table 1, showcases min, max, mean, median and other quantiles of the data. Table 2, showcases numerical columns with significant correlation

- Step 4: Data Split   
Splitting data into train, test and validate.

- Step 5: Fitting on Raw Data    
A set of classifiers with default hyperparamter values are fitted on the raw data and scores are compared.

- Step 6: Feature Transformation   
Features can be transformed by scaling them as that would pertinent for classifiers such as KNN that use distance as a metric. Change of basis for variables that seem necessary can also be done. Only feature scaling has been included in the pipeline.

- Step 7: Feature Selection   
Recursive feature elimination has been used to select relevant set of features.

- Step 8: Fitting on Transformed and Selected Features   
A set of classifiers with default hyperparamter values are fitted on the transformed data with relevant features and scores are compared.

- Step 9: Hyperparamter Optimization
The hyperparamters of the classifiers are optimized using gridsearch and randomized search cross validation.

- Step 10: Fitting on Transfomred and Selected Features with Optimized Hyperparamters
A set of classifiers with optimized hyperparamter values are fitted on the transformed data with relevant features and scores are compared.

- Step 11: Best model
A set of classifiers with optimized hyperparamter values are fitted on the transformed data with relevant features and scores are compared. The model with best f1_score is returned as the best model.


## Usage

- Clone the repository

```
git clone https://github.com/HarjyotKaur/Machine_Learning_Pipeline.git
```

- Choose a data set, make sure the target variable is numerical and binary

- Fill in the following variables

```
path: str
test_size: float
validate_size: float
random_state: int
response:str
classifiers: dictionary
parameters: dictionary
```

## Example

#### Input:

```
# declaring variables

path='data/OnlineNewsPopularity.csv'
test_size=0.2
validate_size=0.2
random_state=500
response='target'

classifiers = {
        'dummy classifer': DummyClassifier(),
        'random forest' : RandomForestClassifier(),
        'naive bayes'   : BernoulliNB(),
        'gradient boosting'  : GradientBoostingClassifier(),
        'logistic regression' : LogisticRegression()
    }

parameters = {

       'dummy classifer' : {
           'parameters': {}
       },
       'random forest' : {
           'parameters': {'max_depth':range(2,20,2), 'n_estimators':range(1,15,2)}
       },
       'naive bayes'   : {
           'parameters': {'alpha':[10**x for x in range(-3, 3)]}
       },
       'gradient boosting'  : {
           'parameters': {'max_depth':range(1,10,2), 'n_estimators':range(1,15,2)}
       },
       'logistic regression' : {
           'parameters': {'C':[10**x for x in range(-3, 3)]}
       }
}

generate_report(path,test_size,validate_size,random_state,response,classifiers,parameters)
```

#### Output:
```
-------------------------------------------------------------------------------------
                              Step 1: Loading Data                                   
-------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------
                              Step 2: Outlier Removal                                
-------------------------------------------------------------------------------------


Outliers Removed 463


-------------------------------------------------------------------------------------
                      Step 3: Descriptive Statstistics                               
-------------------------------------------------------------------------------------


------------ Table 1: Descriptive Statistivs of all numerical columns ---------------
       n_unique_tokens      num_imgs  average_token_length  \
count     39181.000000  39181.000000          39181.000000   
mean          0.530634      4.309691              4.547560   
std           0.136004      7.286509              0.843798   
min           0.000000      0.000000              0.000000   
25%           0.471170      1.000000              4.478528   
50%           0.539007      1.000000              4.663636   
75%           0.607774      4.000000              4.854077   
max           1.000000     54.000000              8.041534   

       global_subjectivity  global_sentiment_polarity  \
count         39181.000000               39181.000000   
mean              0.443017                   0.119691   
std               0.116276                   0.095661   
min               0.000000                  -0.380208   
25%               0.395923                   0.058026   
50%               0.453225                   0.119182   
75%               0.507771                   0.177693   
max               1.000000                   0.655000   

       global_rate_positive_words  global_rate_negative_words  \
count                39181.000000                39181.000000   
mean                     0.039674                    0.016560   
std                      0.017342                    0.010472   
min                      0.000000                    0.000000   
25%                      0.028481                    0.009682   
50%                      0.039062                    0.015385   
75%                      0.050304                    0.021739   
max                      0.136986                    0.081395   

       rate_positive_words  rate_negative_words  avg_positive_polarity  ...  \
count         39181.000000         39181.000000           39181.000000  ...   
mean              0.682583             0.287530               0.353151  ...   
std               0.188942             0.154569               0.103230  ...   
min               0.000000             0.000000               0.000000  ...   
25%               0.600000             0.186441               0.306107  ...   
50%               0.710145             0.280000               0.358416  ...   
75%               0.800000             0.384615               0.410922  ...   
max               1.000000             1.000000               0.872727  ...   

       max_positive_polarity  avg_negative_polarity  min_negative_polarity  \
count           39181.000000           39181.000000           39181.000000   
mean                0.756648              -0.256850              -0.520164   
std                 0.247661               0.122104               0.289747   
min                 0.000000              -0.875000              -1.000000   
25%                 0.600000              -0.326736              -0.700000   
50%                 0.800000              -0.252467              -0.500000   
75%                 1.000000              -0.186111              -0.300000   
max                 1.000000               0.000000               0.000000   

       max_negative_polarity  title_subjectivity  title_sentiment_polarity  \
count           39181.000000        39181.000000              39181.000000   
mean               -0.104396            0.281742                  0.071370   
std                 0.083036            0.323986                  0.264952   
min                -0.666667            0.000000                 -1.000000   
25%                -0.125000            0.000000                  0.000000   
50%                -0.100000            0.142857                  0.000000   
75%                -0.050000            0.500000                  0.143182   
max                 0.000000            1.000000                  1.000000   

       abs_title_subjectivity  abs_title_sentiment_polarity        shares  \
count            39181.000000                  39181.000000  39181.000000   
mean                 0.341986                      0.155603   3001.743039   
std                  0.188805                      0.226010   5313.904316   
min                  0.000000                      0.000000      1.000000   
25%                  0.166667                      0.000000    944.000000   
50%                  0.500000                      0.000000   1400.000000   
75%                  0.500000                      0.250000   2700.000000   
max                  0.500000                      1.000000  73100.000000   

             target  
count  39181.000000  
mean       0.491871  
std        0.499940  
min        0.000000  
25%        0.000000  
50%        0.000000  
75%        1.000000  
max        1.000000  

[8 rows x 21 columns]
-------------------------------------------------------------------------------------


------------ Table 2: Numerical columns with significant correlation ----------------
Empty DataFrame
Columns: [column_x, column_y, corr]
Index: []
-------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------
                                Step 4: Data Split                                   
-------------------------------------------------------------------------------------




-------------------------------------------------------------------------------------
                                Step 5: Fitting on Raw Data                          
-------------------------------------------------------------------------------------


----------------------------------- Table 3: Scores ---------------------------------
                     Training Time  Training Accuracy  Validation Accuracy  \
dummy classifer               0.01              50.29                49.69   
gradient boosting             1.57             100.00               100.00   
logistic regression           0.40              96.39                96.30   
naive bayes                   0.03              52.62                52.27   
random forest                 0.57             100.00               100.00   

                     Test Accuracy  
dummy classifer              49.47  
gradient boosting           100.00  
logistic regression          96.49  
naive bayes                  53.00  
random forest               100.00  
-------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------
                          Step 6: Feature Transformation                             
-------------------------------------------------------------------------------------




-------------------------------------------------------------------------------------
                             Step 7: Feature Selection                               
-------------------------------------------------------------------------------------





The validation error is lowest when features= 7


-------------------------------------------------------------------------------------
                 Step 8: Fitting on Transformed and Selected Features                 
-------------------------------------------------------------------------------------


----------------------------------- Table 4: Scores ---------------------------------
                     Training Time  Training Accuracy  Validation Accuracy  \
dummy classifer               0.00              50.11                49.80   
gradient boosting             1.01             100.00               100.00   
logistic regression           0.16              87.18                86.60   
naive bayes                   0.01              51.09                50.52   
random forest                 0.22             100.00               100.00   

                     Test Accuracy  
dummy classifer              49.84  
gradient boosting           100.00  
logistic regression          86.65  
naive bayes                  51.58  
random forest               100.00  
-------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------
                     Step 9: Hyperparamter Optimization                              
-------------------------------------------------------------------------------------




-------------------------------------------------------------------------------------
  Step 10: Fitting on Transformed and Selected Features with Optimized Hyperparamters
-------------------------------------------------------------------------------------


----------------------------------- Table 5: Scores ---------------------------------
                     Training Time  Training Accuracy  Validation Accuracy  \
dummy classifer               0.01              49.72                50.04   
gradient boosting             0.15             100.00               100.00   
logistic regression           0.18              97.96                97.80   
naive bayes                   0.01              51.09                50.52   
random forest                 0.21             100.00               100.00   

                     Test Accuracy  
dummy classifer              50.25  
gradient boosting           100.00  
logistic regression          97.78  
naive bayes                  51.58  
random forest               100.00  
-------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------
                                 Step 11: Best Model                                 
-------------------------------------------------------------------------------------




Best Classifer for the data is/are: ['gradient boosting', 'random forest']


-------------------------------- Table 5: Final Scores -------------------------------
                     Training Accuracy  Validation Accuracy  Test Accuracy  \
dummy classifer                  50.08                49.03          49.50   
gradient boosting               100.00               100.00         100.00   
logistic regression              97.96                97.80          97.78   
naive bayes                      51.09                50.52          51.58   
random forest                   100.00               100.00         100.00   

                     Precision  Recall  F1 Score  
dummy classifer          49.01   49.70     49.35  
gradient boosting       100.00  100.00    100.00  
logistic regression     100.00   95.46     97.68  
naive bayes              57.14    3.55      6.69  
random forest           100.00  100.00    100.00  
-------------------------------------------------------------------------------------
```
