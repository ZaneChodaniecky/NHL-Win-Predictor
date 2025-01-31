# NHL-Win-Predictor

Created By: Zane Chodaniecky (zane.chodaniecky@gmail.com)

A machine learning model that predicts the outcome of NHL games for the current day's schedule. This binary classifier determines whether the home team will win or lose.

## Features
- **Binary Classification**: Predicts if the home team will win or lose.
- **Real-time Predictions**: Updates for the current day's NHL schedule and provides betting odds
- **Data Preprocessing**: Includes necessary data cleaning and feature engineering steps.
- **Model Evaluation**: Evaluates and reports model performance.

## Data Requirements
- all_teams.csv : Sourced from https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv (Downloaded in preprocessing script)
- shots_YYYY.csv : Sourced from https://peter-tanner.com/moneypuck/downloads/shots_YYYY.zip (Downloaded in preprocessing script)
- NHL_Schedule_2024.csv : Current year (2024) schedule is provided. May need to be recreated custom for future seasons. Sourced from https://media.nhl.com/public/news/18238.
- shots_history.csv : Created using 'Create Shots History.py'. Script will need to be updated with new year file links for future seasons. Sourced from https://moneypuck.com/data.htm
- win_history.csv : File provided here has data through 1/28/2025 and is appended with new data when preprocessing script is run. Pulled from official NHL api https://api-web.nhle.com/v1/wsc/game-story/
- goalie_history.csv : File provided had data through 1/28/2025 and is appended with new data when preprocessing script is run. Calculated from shots_YYYY.csv
- team_abbreviations.csv : This will need to be updated if a new team is added to the league. Also will need to be updated in 2025 when Utah selects a permanent team name.

### Notes Users
- This is created for 2024 season
- To start new season, URL for shots_2024.zip must be updated in 'NHL Win Classification - PreProcessing.py'
- To start new season, file location for for shots_2024.csv must be updated in 'NHL Win Classification - PreProcessing.py'
- To start new season, you will need to source and generate a new 'NHL_Schedule_YYYY.csv' file and format like the 2024 file. Not sure how to future-proof this step.
- At the conclusion of a season, please run the 'NHL Win Classification - PreProcessing.py' a final time to fully populate 'win_history.csv' and 'goalie_history.csv' for the season.
- The data that is being used is the average statistics for a given feature over the past X games in the current season (set to 7 right now) so early season predictions may have lower accuracy until 7 games in.
- Odds are sourced from 'the-odds-api.com'. This script contains my API keys, you can generate your own for free if you wish.
  

### Prerequisites
- This project requires Python 3.x and the following libraries:
- catboost>=1.2.7
- matplotlib>=3.7.2
- numpy>=1.24.3
- pandas>=2.0.3
- requests>=2.31.0
- scikit-learn>=1.5.2
- scikit-optimize>=0.10.2
- seaborn>=0.12.2
- statsmodels>=0.14.4
- xgboost>=2.1.3

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/ZaneChodaniecky/NHL-Win-Predictor.git
cd NHL-Win-Predictor
pip install -r requirements.txt
python run.py
