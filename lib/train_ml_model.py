#Boosting algorithms combine multiple low accuracy(or weak) models to create a high accuracy(or strong) models.
#Ensemble learning methods are meta-algorithms that combine several machine learning methods into a single predictive model to increase performance. Ensemble methods can decrease variance using bagging approach, bias using a boosting approach, or improve predictions using stacking approach.
#Boosting algorithms are a set of the low accurate classifier to create a highly accurate classifier. Low accuracy classifier (or weak classifier) offers the accuracy better than the flipping of a coin. Highly accurate classifier( or strong classifier) offer error rate close to 0. Boosting algorithm can track the model who failed the accurate prediction. Boosting algorithms are less affected by the overfitting problem.

#The basic concept behind Adaboost is to set the weights of classifiers and training the data sample in each iteration such that it ensures the accurate predictions of unusual observations.

#Adaboost should meet two conditions:

#The classifier should be trained interactively on various weighed training examples.
#In each iteration, it tries to provide an excellent fit for these examples by minimizing #training error.

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


def train_machine_learning_model(X_train,y_train,X_test,y_test,model_name,base_estimator=None):
    if(model_name=='adaboost'):
        # Create adaboost classifer object
        #important parameters: #base_estimator, n_estimators, and learning_rate.
        #base_estimator: It is a weak learner used to train the model. It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
        #n_estimators: Number of weak learners to train iteratively.
        #learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.

        #pros = AdaBoost is not prone to overfitting
        #cons = sensitive to noise data.
        #       It is highly affected by outliers because it tries to fit each point perfectly. 
        #       AdaBoost is slower compared to XGBoost.
        abc = AdaBoostClassifier(n_estimators=50,
                            learning_rate=1,base_estimator=base_estimator)
        # Train Adaboost Classifer
        model = abc.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elif(model_name=='xgboost'):
        #Extreme Gradient Boosting
        #pros - Speed and performance, Core algorithm is parallelizable,Wide variety of tuning parameters
        #important parameters:
            #learning_rate: range [0,1]
            #max_depth : determines how deeply each tree is allowed to grow
            #subsample: percentage of samples used per tree.Low values can lead to underfitting
            #colsample_bytree: percentage of features used per tree. High values can lead to overfitting
            #n_estimators: number of trees
            #objective: determines the loss function to be used:
                #reg:linear= regression
                #reg:logistic= for classification
                #binary:logistic for classification problems with probability
            #gamma-controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits.
            #alpha: L1 regularization on leaf weights. A large value leads to more regularization.
            #lambda: L2 regularization on leaf weights and is smoother than L1 regularization.
        data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
        xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
        xg_reg.fit(X_train,y_train)
        y_pred = xg_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elif(model_name=="logistic_regression"):
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        # define the model evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model and collect the scores
        model.fit(X_train,y_train)
        y_pred=model.predict(X_train)
    return rmse,metrics.confusion_matrix(y_test,y_pred),metrics.classification_report(y_test,y_pred)
        

def train_catboost(X_train,y_train):
    #pros-Performance,Handling Categorical features automatically,Robust,EasytoUse
    categorical_features_indices = ['I', 'II', 'III', 'IV', 'V']
    model = CatBoostClassifier(
                        iterations=1000,
                        random_seed=63,
                        learning_rate=0.01,
                        depth = 7,
                        eval_metric = 'AUC',
                        custom_loss=['AUC','F1','Precision','Recall'],
                        early_stopping_rounds = 100,
                        nan_mode= 'Min',
                        class_weights={0: 1, 1: 10},
                        l2_leaf_reg = 2,
                        boosting_type = 'Ordered',
                        train_dir = "base_model")
                        #auto_class_weights)
    result = model.fit(X_train, y_train,cat_features=categorical_features_indices,plot=True)
    return result         

def train_fold_cross_validation_xgboost(X_train,y_train):
    #important parameters:
            #num_boost_round: denotes the number of trees you build 
            #metrics: tells the evaluation metrics to be watched during CV
            #as_pandas: to return the results in a pandas DataFrame.
            #early_stopping_rounds: finishes training of the model early if the hold-out metric ("rmse" in our case) does not improve for a given number of rounds.
            #seed: for reproducibility of results.
    params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
            'max_depth': 5, 'alpha': 10}
    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
            num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    return cv_results



    
    
