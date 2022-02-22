## Description
The Recommender System Blueprint is a set of algorithms that generates responses for recommending similar items to users. The responses are based on their choices from amongst a set list of items, and are specifically tailored to not repeat items that have already been rated or viewed by the user.
The Blueprint requires user data in the form of user’s choice on the current items, for the model to learn the user choices and predict recommendations on all the items, for each user. The predictions are directly based on the scores, which are nothing but predicted ratings, provided by the model, for all the items rather than just the ones which the user has already seen/rated.

## Who Is This For?
Intermediate Users (Business analysts, Marketing Associate, Data Scientists), Business owners (Retail, Entertainment, Academic) etc

## Modes of Operation
The blueprint requires user data to train the models on ratings data on a list of items and predict recommendations based on the scores given by the model. It cannot extrapolate its extrapolate its recommendations on unknown users
(this  functionality is a potential future advancement) Recommender System blueprint can be used in two methods:
* Get a recommendation on a user existing in the common datasets
* Train the model on a completely new dataset and then give recommendations on that

In both methods, the hyper-parameters can be customized and the user can choose to select the model too, for the final recommendation,  in case they are advanced users.

## Data

| user_id | item_id | rating |
|---------|---------|--------|
| 0       | 0       | 1      |
| 0       | 1       | 3      |
| 1       | 0       | 4      |

There are a few constraints on the data:
* The rating value is supposed to be within 0-10. The maximum rating can be any number below 10 (5 etc).
  * The models will may work for ratings beyond 10 as well, but with a reduced accuracy.
  * The user can standardize the data however, the predictions accuracy will be redu
    ced.
* The data magnitude has to be within certain limits:
  * The number of users and number of items have to be under 10,000 each for the sake of processing time, unless the user has a significantly larger than average compute.

## What is in the package?
 
The Recommender System Blueprint consists of 6 algorithms of different types of collaborative filtering: 
* Matrix Factorization
* Regularized Matrix Factorization
* Singular Value Decomposition
* Singular Value Decomposition ++
* Alternating Least Squares
* Non-Negative Matrix Factorization. 

They differ mostly in their cost function values. The blueprint compares between all 6 models and extracts the best one out of them based on a composite evaluation metric.

## Evaluation Metrics

Composite Metric = (Rec_Rel_Count * (1/100) * 0.1) + (1/RMSE) * 0.4 + (Recall @ K * 0.25) + (Precision @ K *0.25)

where: </br>
`Rec_Rel_Coun`t  = Recommended and Relevant Choices in Test Dataset</br>
`RMSE`                 = Root Mean Square Error of Test Dataset</br>
`Recall @ K`         = Recomm. Choices Count amongst top K choices (by predicted score)/ Total Relevant choices in Test Dataset</br>
`Precision @ K`     = Recomm. Choices  Count amongst top K choices (by pred. score) / Total Recomm. Choices in Test Dataset 

The reason for such a metric is because one metric alone can bias the results in case of very low or very high predicted scores. High precision may mean extremely few predictions in total, while high recall alone may mean that predictions are higher than average for all items. To remove these biases, such a metric has been designed.

RMSE will move the final choice towards accuracy. Precision and Recall will balance the final choice as not being too skewed towards a higher/lower score. Rel_Rec_Count will move the final choice towards obtaining a higher absolute number of final predictions.

### Read More
[Sentiment Analysis: Concept, Analysis and Applications](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17)

[Build and Deploy IMDB NLP model](https://app.cnvrg.io/docs/tutorials/build_and_deploy_imdb.html)
