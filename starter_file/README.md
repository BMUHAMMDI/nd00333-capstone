

# Capstone Project - Azure Machine Learning Engineer 
   ## Heart Failure Predection 
This project will focus on predicting heart disease using different tools available in AzureML framework. This is an opportunity to use the knowledge we have obtained from this Nanodegree to solve this problem. In this project, we will create two models: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.
  
## Dataset

### Overview
The data used for training is the Heart Disease UCI downloaded from Kaggle. This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. This is a classification problem, with input features as a variety of parameters, and the target variable DEATH_EVENT which is a binary variable, predicting whether heart disease is present or not (1=yes, 0=no). The input features along with meanings, measurement units, and intervals of each feature as described below:

![Capture88](https://user-images.githubusercontent.com/52258731/104820811-7c3a0980-5848-11eb-8315-5ab393c80209.JPG)

### Task
We used two different Machine Learning approaches: Auto ML and Hyperparameter tuning to classify whether a person is suffering from heart disease or not. We also determined the best model, deployed and used it to see how the model is working.

### Access
We downloaded the dataset from Kaggle and uploaded it in this github repository. Then, we registered the dataset in Azure ML workspace using its GitHub URL. The dataset is loaded using TabularDatasetFactory() function available in azure-ml.

## Automated ML
Automated machine learning is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.
 We used AutoMLConfig class to specify this task's parameters as follows:
 
 - task='classification': it's the type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of ML problem   	to solve.
 - compute_target='cpu-cluster': the Azure Machine Learning compute target to run the Automated ML experiment on.cpu-cluster has been created at the c.
 - primary_metric = 'accuracy':it's the metric that Automated ML will optimize for model selection.
 - training_data= train_data: it's the training data to be used within the experiment. It should contain both training features and a label column. this   	dataset has been created at the beginning as well.
 - label_column_name ='DEATH_EVENT':it's the name of the label column used in the training_data. 
 - enable_onnx_compatible_models=True : it decides whether to enable or disable enforcing the ONNX-compatible models. The default is False.
 - featurization='auto': featurizationConfig Indicator for whether featurization step should be done automatically or not, or whether customized 	   	featurization should be used. 
  - enable_early_stopping= True: to decide whether to enable early termination if the score is not improving in the short term. The default is False.
  - **automl_settings (Here, we specified "experiment_timeout_minutes" to be 25 minutes,"max_concurrent_iterations" set to 4, and "iterations" set to 20)

### Results

The best model obtained through AutoML is VotingEnsemble model which has an accuracy of 90% as shown here:

![Picture1](https://user-images.githubusercontent.com/52258731/104819266-72f76f80-583d-11eb-9a92-d0be147fc97f.png)

Parameters (we get the parameters details from get_output() method ):
 - min_child_weight=0,
 - min_split_gain=0.42105263157894735,
 - n_estimators=25,
 - n_jobs=1,num_leaves=191,
 - objective=None,
 - random_state=None,
 - reg_alpha=0.15789473684210525,
 - reg_lambda=0,
 - silent=True,
 - subsample=1,
 - subsample_for_bin=200000,
 - subsample_freq=0,
 - verbose=-10

The accuarcy could be improved by increasing the experiment_timeout_minutes parameter as will run the automl for longer which may increase its accuracy. Also, it could be improved by dropping one or two features that might not be as helpful. 

Following are screenshots of the RunDetails widget as well as a screenshot of the best model trained with it's parameters:

### Run Details

![Picture2](https://user-images.githubusercontent.com/52258731/104819295-97534c00-583d-11eb-8455-3c598e3d1d76.png)
![Picture3](https://user-images.githubusercontent.com/52258731/104819321-c1a50980-583d-11eb-9e5a-09603c129bd8.png)

### Best model and run_id

![Picture4](https://user-images.githubusercontent.com/52258731/104819354-ef8a4e00-583d-11eb-9a54-be389fb71301.png)
![Picture6](https://user-images.githubusercontent.com/52258731/104819364-00d35a80-583e-11eb-9953-d39eb5cc6aa6.png)


## Hyperparameter Tuning

We have created a logistic-regression model from scikit-learn and used hyperparameters 'Inverse of Regularization Strength' and 'Maximum number of iterations to converge' for parameter sampling. The sampling method used is RandomSampling which supports both discrete and continuous values. In this sampling method, the values are selected randomly from a defined search space.Early stopping policy used is Bandit Policy. Bandit Policy is based on slack factor/slack amount and evaluation interval. This policy will terminate runs whose primary metric is not within the specified slac factor/slack amount.

Parameters used for ramdom sampling (ps):

   - C: The inverse of the reqularization strength. {'--C': choice(0.6,1,1.5)}
   - max_iter: Maximum number of iterations to converge. {'--max_iter': choice(10,20,30,40)}

We configured the hyperdrive run as follows:

 - estimator=est: an estimator that will be called with sampled hyperparameters.
 - hyperparameter_sampling=param_sampling: the hyperparameter sampling space.
 - policy=early_termination_policy: States the early termination policy to be used.
 - primary_metric_name='Accuracy': it's the name of the primary metric reported by the experiment runs.
 - primary_metric_goal=PrimaryMetricGoal.MAXIMIZE:this parameter determines if the primary metric is to be minimized or maximized when evaluating runs.
 - max_total_runs=20: this is the maximum total number of runs to create. It may be fewer runs when the sample space is smaller than this value.
 - max_concurrent_runs=4: the maximum number of runs to execute concurrently.If None, all runs are launched in parallel.

### Results

Hyperdrive run produced a best model with 84% accuracy. Parameters of the model were:C = 1 and max_iter = 40

The accuracy could be improved by choosing a different model like decision tree,randomforests,etc or by choosing a different parameter sampling policy.

Following are screenshots of the RunDetails widget as well as a screenshot of the best model trained with it's parameters:

### Run Details

![Picture7](https://user-images.githubusercontent.com/52258731/104819380-22344680-583e-11eb-9e57-5479d0a8827a.png)
![Picture8](https://user-images.githubusercontent.com/52258731/104819385-31b38f80-583e-11eb-9d15-55c4a5af68d6.png)

### Best model and run_id

![Picture9](https://user-images.githubusercontent.com/52258731/104819411-59a2f300-583e-11eb-8cf6-a6397a251277.png)

## Model Deployment 
We decide to deploy the best Auto-ml Model. The best model is registered and then deployed as a Webservice on Azure Container Instances.The model endpoint was queried by sending a post request to the model over the REST URL.

Following are screenshots of the successfully deployed model and the service request:

### Deployed Mode

![Picture10](https://user-images.githubusercontent.com/52258731/104819428-83f4b080-583e-11eb-9240-e7da2323b1dc.png)

### Service Request
Here is a shot of the inference request sent to the deployed model and how it returned the value [1], which means the model is running and our test patient has a heart disease.

![Capture11](https://user-images.githubusercontent.com/52258731/104838984-6de30080-58cf-11eb-9b3d-9e5a359b870d.JPG)

## Screen Recording

https://www.youtube.com/watch?v=tFuW24AZFXw

