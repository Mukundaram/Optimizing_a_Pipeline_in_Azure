# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data used for this project is a bank marketing campaign dataset using which we can predict whether a given client would subscribe to a term deposit or not. It contains the clients' personal and demographic details, social and economic context attributes along with the marketing campaigns details which were based on phone calls.

Two different experminets were conducted - the first experiment was performed using **Hyperdrive** to search for the best hyperparameters while the second was was performed using **AutoML**. 

The best model was selected by using **Accuracy** as the primary metric of comparison. The best model,produced by the **AutoML** run, used **VotingEnsemble** technique and the underlying classifier was **LightGBM**. The **Accuracy** of the model is **91.76%**.

## Scikit-learn Pipeline
The dataset is initially cleaned and prepared for modelling. Some of the cleaning steps were converting categorical features to numerical features using one hot encoding or encoding them as binary encoded features. After cleaning the dataset and preparing the input columns, the data is split into train and test datasets. **30%** of the data was used as the testing dataset.

**Logistic Regression** was selected as the model of choice for preforming the **classification task** and **Accuracy** was selected as the metric to optimize.

Furthermore, for **hyperparameter tuning** using **Hyperdrive**, the hyparameter space was defined for **C** and **max_iter** parameters. **RandomParameterSampling** was used to search over different sets of hyperparameters to maximize our primary metric. 

The **regularization hyperparameter (C)** helps to control or prevent model overfitting. For small values of C, we increase the regularization strength which will create simple models which underfit the data. For big values of C, we low the power of regularization which imples the model is allowed to increase it's complexity, and therefore, overfit the data. The **maximum iterations hyperparameter** helps us to find out the ideal number of iterations for the model to converge.

The **bandit termination policy** helps us to avoid unnecessary runs by stopping the iteration early whenever the primary metric falls outside the slack factor threshold. This will further ensure that every run will give us a better metric than the previous one.

The best model was generated using the **regularization strength(C) as 0.01** and the **maximum number of iterations(max_iter) as 25**.  The **Accuracy** of this model was **91.34%**.

![Best model - Hyperdrive](https://github.com/Mukundaram/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Ouput_images/Hyperdrive_best_run.PNG "Best model overview - Hyperdrive")

## AutoML
For the AutoML run, we first configure the run by selecting the dataset, column to predict, the metric to optimize (Accuracy) and the type of task to perform which is classfication in this case along with other parameters like maximum iterations to perform and experiment timeout minutes. We then submit the experiment method and pass the run configuration.. 

In this case, the **VotingEnsemble Algorithm** turned out to be the best model with the underlying classifier being **LightGBM**. It yielded an **Accuracy** of **91.76%**.

![Best model - AutoML](https://github.com/Mukundaram/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Ouput_images/AutoML_best_model_metrics.PNG "Best model overview - AutoML")

## Pipeline comparison
AutoML gave us the best model for this dataset although the **Accuracy** of two best models using AutoML and Hyperdrive was almost the same. The dfference in accuracy was 0.42%.
The **Logistic regression** model is compartively simpler in nature than the **VotingEnsemble** which combines the predction of multilple weak learners to imrove the models' ability. 

Moreover, Hyperdrive was configured to use only Logistic Regression as the model of choice whereas AutoML had the advantage of trying out various algorithms including an ensemble of the different models as well to produce a robust model. Therefore, AutoML had the chance to explore more and produce a better model than Hyperdrive.

## Future work
**Class Imbalance**: The dataset was imabalanced and this was highlighted in the AutoML run details. This can be handled accordingly by using various data sampling techniques like oversampling and undersampling or a combination of different techniques.

**Model evaluation metric**: Since the dataset is imbalanced, **Accuracy** may not be the right choice to decide thebest model. Metrics like **Recall and F1 Score** are better choices for imabalanced datasets and will help us choose the optimal model for the same.

**Different configurations for Hyperdrive**: Hyperdrive can be tried by using different models/algorithms apart from Logistic Regression and by defining a more exhaustive hyperparameter space i.e. specify more number of hyperparamters with multiple values to choose from. Also, Bayesian Parameter Sampling can be used to achieve quick model convergence.

## Proof of cluster clean up
After running the experiments, the compute cluster was deleted.

[Cluster cleanup](https://github.com/Mukundaram/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Ouput_images/Cluster_cleanup.PNG)
