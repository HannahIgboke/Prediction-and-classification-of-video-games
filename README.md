# Prediction and classification of video games 🎮

<p align="left">
    <img width="800" src="https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/blob/main/Notebooks/Images/video_games.jpg" alt="Video games">
</p>

# Project Overview
The online gaming industry experiences unpredictable changes in sales and performance metrics due to the continuous advent and improvement of video games. This project takes a deep dive into historical data on video game sales to:

- Understand what the gaming industry market in the last three decades has been like
- Identify key features influencing the performance of these games globally and regionally.
- Predict global sales based on relevant properties
- Provide a solution to the question: what game feature combinations will turn in high or low sales?

The project workflow below seeks to address these pain points.

# Workflow
- Data collection
- Data preparartion
  - Missing data treatment
  - Feature engineering
- Exploratory data analysis
- Impact of features
- Prediction of global sales
- Classifier for sales category
- Model deployment and hosting



# Packages and tools used
Some tools and packages used in the course of this project include:

- Pandas and numpy
- KNNImputer
- Seaborn, and matplotlib
- Scikit
- Joblib
- Streamlit


# Data
The data used here was obtained from [Kaggle](https://www.kaggle.com/datasets/ibriiee/video-games-sales-dataset-2022-updated-extra-feat/data). It contains information about video game sales worldwide, including factors such as critic and user reviews, genre, platform, and more. Note that sales is in millions.

| Column Name      | Description                                            |
|------------------|--------------------------------------------------------|
| Name             | The name of the video game.                            |
| Platform         | The platform on which the game was released.           |
| Year_of_Release  | The year in which the game was released.               |
| Genre            | The genre of the video game.                           |
| Publisher        | The company responsible for publishing the game.       |
| NA_Sales         | The sales of the game in North America.                |
| EU_Sales         | The sales of the game in Europe.                       |
| JP_Sales         | The sales of the game in Japan.                        |
| Other_Sales      | The sales of the game in other regions.                |
| Global_Sales     | The total sales of the game across the world.          |
| Critic_Score     | The average score given to the game by professional critics. |
| Critic_Count     | The number of critics who reviewed the game.           |
| User_Score       | The average score given to the game by users.          |
| User_Count       | The number of users who reviewed the game.             |
| Developer        | The company responsible for developing the game.       |
| Rating           | The rating of the game.                                |


P.S: All accompanying codes for the next steps are contained in the respective notebooks to be linked. This is done to ensure the brevity and conciseness of this readme. This proceeding steps below will contain the thought process for them, relevant results and codes if necessary.



# Data preparation
Following best practices, a copy of the dataset was made after importation and all analysis were carried out on that copy. Conducting an initial exploratory data analysis revealed:
- No duplicate rows
- Inappropriate data types for some columns
- Missing data as seen below

![image](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/assets/116895464/1206757d-6859-4763-9223-5e21f38b212d)

- Summary statistics for numerical columns
- Unique values for categorical columns

## Tackling missing data
Upon further analysis, three categories of missing data was observed. Each were treated differently.

1. Missing Completely at Random(MCAR)
Where the probability of a data point missing is entirely unrelated to any other observed/unobserved data. The name and genre columns fell into this category. Since the number of missing values for this were negligible, they were therefore dropped from the dataset

2. Categorical columns like Publisher, rating, etc
The NaN was replaced with "missing" to indicate unavailability of relevant data

3. Missing at Random(MAR)
This applied to the missing values in the numerical columns where missingness is not completely random but can be explained by some other known information. These rows cannot dropped as that will lead to gross information loss thereby impacting the efficiency of our model in the future. To handle this, I used a a multivariate approach - the KNNImputer with k=5 nearest neighbors which allows the imputer to find the 5 most similar rows in the dataset and make imputations for each.

After proper handling of all the cases aforementioned, we have this:

![image](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/assets/116895464/31dd3732-5ab7-4ab5-ab1d-49c2a4c0f5bd)

## Feature engineering

This involved creating new features based on already available information. I created a new feature called release_era that groups the release year of games into three eras - pre-2000s, 2000-2010, and post-2010. This was created to enable me perform some group level analysis during the EDA process.


# Exploratory Data Analysis
The notebook for a detailed breakdown of the EDA process including univariate and bivariate analysis can be found [here](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/blob/main/Notebooks/Exploratory%20Data%20Analysis.ipynb). Here, I present some questions I answered and insights revealed from this stage in my analysis.

1. What have the sales through the years been like regionally?

![image](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/assets/116895464/54f3c83f-2587-487f-af46-90634b573321)

Generally, sales spiked for all regions in 1995.

In North America, there was a wave of fluctuations from 1980 till 1995 when it picked up and rose steadily with a few dips here and there. This reached peak sales of 350 million copies. This was not sustained though, the sales began dwindling through the years till 2016. The sales in all regions were low compared to a decade before that. Sales in Europe and Japan follow a similar pattern. For sales in "other" regions, we see a relatively steady growth from less than one million sales to its highest point around 70 million and then a decline.


2.  Is there a correlation between critic scores and user scores? Do they tend to agree or disagree?

![image](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/assets/116895464/390aeebb-a115-45c3-805f-c72bfd08dbe6)

Based on the correlation plot and a heatmap created during the [analysis](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/blob/main/Notebooks/Exploratory%20Data%20Analysis.ipynb), there is a moderate positive linear association (0.5) between the user score and the critic score. What does this mean?

This means that there is a noticeable trend between the critic scores and the user scores, even though it is not a perfect relationship. It also tells us that professional critics and users tend to agree to some extent on their assessment of video games. Therefore, stakeholders need to consider both critic and user scores when making decisions about game development and marketing. 


Want to see more? Check out the complete EDA process [here](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/blob/main/Notebooks/Exploratory%20Data%20Analysis.ipynb).


# Impact of features on regional sales




# Prediction of global sales



# Classifier: what game feature combinations will turn in high or low sales?




# Model deployment and hosting




# Conclusion




# Future work 




# License







