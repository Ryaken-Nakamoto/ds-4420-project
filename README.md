# Predicting Point Differentials in NBA Game Outcomes

## Introduction/Abstract

Predicting the outcome of basketball games is a well-studied problem in sports analytics. The National Basketball Association (NBA) is an ideal setting to apply machine learning methods to forecast game outcomes. Existing methods have primarily framed game outcomes as a binary classification problem (predicting wins and losses). These methods have achieved meaningful accuracy, but leave out the continuous margin of victory and the uncertainty of the predictions. We present a two-model framework for NBA game predictions, where we use 10-game performance statistics to predict the point differentials of a game. We trained a manually-implemented Multilayer Perceptron (MLP) model to predict the point differentials, and a Bayesian Linear Regression model to produce posterior predictive distributions with 95% credible intervals for each game.

## Motivation and Data

These ML models are inspired by previous papers that attempt to predict NBA game outcomes with ML (Horvat et al). Instead of predicting win-loss, we shift to point differentials.

All of the training data came from basketball-reference.com, downloaded through an intermediary Kaggle page. These stats were transformed into 10-game rolling averages. From there, we calculated a differential between the home and away teams for these features.

Note that for both of these models, it is only possible to predict the very next game between any two given teams. Because of this, our test set is capped at 30*29 = 870 examples. We trained the model on NBA seasons 2015-2019, and tested it with performance during the 2020, using the very first occurrence of each home-team and away-team permutation. Certain features are strongly correlated with each other, such as FT% and Points, adding inherent multicollinearity.

## Methodology

We employ two different models to predict point differentials. A Bayesian Linear Regression using a Gaussian posterior/prior, and an MLP. For the Bayesian model, we use consistent hyperparameters of a mean of zero and a standard deviation of one. Both models take in the same input data. The MLP model consists of an input layer of size 6 and two hidden layers of size 64 and 16 with ReLU activation functions. The weights are initialized from a normal distribution with He initialization, in order to prevent gradient divergence. The MLP is trained on only the rolling performance differentials, as the home/away columns caused the model to overfit.

## Results

The Bayesian linear regression model had, on average, 90% of the inferences within 95% of the distribution. This implies that there exists high variance within the games, which we are not accounting for. Furthermore, the posterior standard deviation is 13, meaning there is relative low confidence in the output of the model. Most games in the NBA lie within the 13 point differential.

The MLP achieved an MSE of 220.29, RMSE of 14.84, and R^2 score of 0.0518. This outperforms the naive baseline and OLS baseline, which respectively have MSE values of 237.30 and 242.11, and RMSE values of 15.40 and 15.56. However, the MLP predicts score differentials closer to the mean compared to the actual differentials. The R^2 score indicates that the rolling differential features have a limited explanatory power for score margins in individual games. The bias term of the output layer, 0.3932, while lower than the training mean of 2.66 points, indicates that there remains a home court advantage.

Other notable results:
* Rebound differentials are the strongest indicators of a positive point differential
* 3FG% is not a strong indicator of positive point differentials

## Discussion/Future Work

Our study found that predicting NBA point differentials is constrained by high variance and multicollinearity in the rolling differential features. We were able to organically capture the general indicators of good performance, such as playing at home, higher FG%s, assists, and rebounds. However, the assumptions of using linear regression were violated, as consecutive games from the same team share 90\% of the same data. We decided to use these features with this approach since this is what Horvat et al. did in their paper. The MLP’s marginal improvement over OLS on this dataset suggests nonlinear interactions among the features (although this may be noise). Both models agree that the primary bottleneck is feature informativeness rather than the capacity of the model.

We recommend that future iterations of this approach prioritize richer feature engineering, including team-specific interaction terms and a decaying effect of training data based on its age (for example, the Bayesian model disfavors the Lakers due to poor performance in 2015-2016 before the LeBron era). Creating features to factor in star player trades would also be crucial. We also recommend that a sequential architecture, such as an LSTM, be implemented for the neural network model to capture the temporal structure of the data.
