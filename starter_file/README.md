

# Capstone Project - Azure Machine Learning Engineer 
   ## Heart Failure Predection 
This project will focus on predicting heart disease using different tools available in AzureML framework. This is an opportunity to use the knowledge we have obtained from this Nanodegree to solve an interesting problem. In this project, you will create two models: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.
  

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
The data used for training is the Heart Disease UCI downloaded from Kaggle. This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. This is a classification problem, with input features as a variety of parameters, and the target variable DEATH_EVENT which is a binary variable, predicting whether heart disease is present or not (1=yes, 0=no). The input features along with meanings, measurement units, and intervals of each feature as described below:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6998201/table/Tab1/?report=objectonly
Feature	Explanation	Measurement	Range
Age	Age of the patient	Years	[40,..., 95] 
Anaemia	Decrease of red blood cells or hemoglobin	Boolean	0, 1
High blood pressure	If a patient has hypertension	Boolean	0, 1
Creatinine phosphokinase	Level of the CPK enzyme in the blood	mcg/L	[23,..., 7861]
(CPK)			
Diabetes	If the patient has diabetes	Boolean	0, 1
Ejection fraction	Percentage of blood leaving	Percentage	[14,..., 80]
	the heart at each contraction		
Sex	Woman or man	Binary	0, 1
Platelets	Platelets in the blood	kiloplatelets/mL	[25.01,..., 850.00]
Serum creatinine	Level of creatinine in the blood	mg/dL	[0.50,..., 9.40]
Serum sodium	Level of sodium in the blood	mEq/L	[114,..., 148]
Smoking	If the patient smokes	Boolean	0, 1
Time	Follow-up period	Days	[4,...,285]



### Task
We used two different Machine Learning approaches: Auto ML and Hyperparameter tuning to classify whether a person is suffering from heart disease or not. We also determined the best model, deployed and used it to see how the model is working.

### Access
We downloaded the dataset from Kaggle and uploaded it in this github repository. Then, we registered the dataset in Azure ML workspace using its GitHub URL. The dataset is loaded using TabularDatasetFactory() function available in azure-ml.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
