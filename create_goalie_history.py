# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:51:59 2024

@author: zchodan
"""

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
import os
import socket


shotsHistoryPath = r'Data\shots_history.csv'
shotsCurrentYearPath = r'Data\shots_2024.csv'

df_shots_history = pd.read_csv(shotsHistoryPath)
df_shots_current_year = pd.read_csv(shotsCurrentYearPath)

 
# Filter for columns that we want
keep_columns = ['season','isPlayoffGame','game_id','team','homeTeamCode','awayTeamCode','isHomeTeam','goalieIdForShot','goalieNameForShot',
                'goal','shotWasOnGoal','time','shotOnEmptyNet'
                ]

# Clean up dataframe
df_shots_history = df_shots_history[keep_columns].copy()  
df_shots_history = df_shots_history.dropna(subset=['goalieIdForShot']) # Drop empty net goals
df_shots_history['goalieIdForShot'] = df_shots_history['goalieIdForShot'].astype(str).str.replace('.0','',regex=False) # Convert goalie ID to str
df_shots_history['isHomeTeam'] = df_shots_history['isHomeTeam'].astype(str).str.replace('.0','',regex=False) # Convert isGoalieTeamHome to str
df_shots_history['shotWasOnGoal'] = df_shots_history['shotWasOnGoal'].astype(int) # Convert shotWasOnGoal to int

# Clean up datafram
df_shots_current_year = df_shots_current_year[keep_columns].copy() 
df_shots_current_year = df_shots_current_year.dropna(subset=['goalieIdForShot'])  # Drop empty net goals
df_shots_current_year['goalieIdForShot'] = df_shots_current_year['goalieIdForShot'].astype(str).str.replace('.0','',regex=False) # Convert goalie ID to str
df_shots_current_year['isHomeTeam'] = df_shots_current_year['isHomeTeam'].astype(str).str.replace('.0','',regex=False) # Convert isGoalieTeamHome to str
df_shots_current_year['shotWasOnGoal'] = df_shots_current_year['shotWasOnGoal'].astype(int) # Convert shotWasOnGoal to int

df_Combined = pd.concat([df_shots_history,df_shots_current_year], ignore_index=True)


# Remove missed shots playoff games to redce dataframe workload
df_Combined = df_Combined.query("isPlayoffGame == 0 & shotWasOnGoal == 1 & season >= 2015")  

# Rename fields to make data less confusing
# This is shot data, but we are looking from the goalies perspective
df_Combined.rename(columns={'goalieIdForShot':'goalieId'}, inplace=True)
df_Combined.rename(columns={'goalieNameForShot':'goalieName'}, inplace=True)


# Create fullGameId
df_Combined['fullGameId'] = df_Combined['season'].astype(str) + df_Combined['isPlayoffGame'].astype(str) + df_Combined['game_id'].astype(str)


# Create goalie team column based on is shooter home team
df_Combined['isGoalieTeamHome'] = np.where(df_Combined['isHomeTeam'] == '1', '0','1')
df_Combined.drop(columns=['isGoalieTeamHome'])
df_Combined['goalieTeam'] = np.where(df_Combined['isGoalieTeamHome'] == '1', df_Combined['homeTeamCode'],df_Combined['awayTeamCode'])



# Find which goalie was in net at the end of the game, in case a goalie got pulled during the game
# Group by 'GameId' and 'Team', then find the index of the row with the max 'Time' in each group
df_last_in_net = df_Combined.query("shotOnEmptyNet == 0 & goalieName.notna()").copy() 
# Ensure the index is preserved/reset if necessary
df_last_in_net = df_last_in_net.reset_index(drop=True)
max_time_indices = df_last_in_net.groupby(['fullGameId', 'goalieTeam'])['time'].idxmax()

# Create a new DataFrame from the rows with the max 'Time' in each group
df_max_time_goalies = df_last_in_net.loc[max_time_indices, ['fullGameId', 'goalieTeam', 'goalieId']]


# Merge original DataFrame with max_time_goalies on 'GameId', 'Team', and 'GoalieId'
df_Combined = df_Combined.merge(df_max_time_goalies.assign(lastGoalieInNet=1), on=['fullGameId', 'goalieTeam', 'goalieId'], how='left')

# Fill NaN values in 'Winner' with 0 for non-matching rows
df_Combined['lastGoalieInNet'] = df_Combined['lastGoalieInNet'].fillna(0).astype(int)


##### CALCULATE GAA #####

# Remove empty net goals since we're calculating GAA
df_no_empty_net = df_Combined.query("shotOnEmptyNet == 0 & goalieName.notna() & goalieId != 0")   

# Group by 'game_id' and 'goalie_id' then sum the 'goal' column to get total goals for each game/goalie
df_goals_per_game = df_no_empty_net.groupby(['fullGameId','goalieId','goalieName','season','goalieTeam','isGoalieTeamHome','lastGoalieInNet'])['goal'].sum().reset_index()

# Rename the column to make it clear that it's the total number of goals
df_goals_per_game.rename(columns={'goal': 'totalGameGoals'}, inplace=True)

  
# Step 1: Create a cumulative count of games for each goalie in each season
df_goals_per_game['cumulativeGames'] = (
df_goals_per_game
.groupby(['season', 'goalieId'])
.cumcount() + 1  # +1 to start count from 1 instead of 0
)

# Step 2: Calculate the rolling average with a variable window based on cumulative games
df_goals_per_game['goalieIdSeasonGAA'] = (
df_goals_per_game
.groupby(['season', 'goalieId'])
['totalGameGoals']
.transform(lambda x: x.rolling(window=len(x), min_periods=1).mean())
)

# Step 3: Calculate the rolling average with a variable window based on cumulative games
df_goals_per_game['goalieIdSeasonGAA'] = (
df_goals_per_game
.groupby(['season', 'goalieId'])
['totalGameGoals']
.transform(lambda x: x.rolling(window=len(x), min_periods=1).mean())
)


##### CALCULATE SV% #####

# Using goals table, Create a cumulative count of games for each goalie in each season
df_goals_per_game['cumulativeGoals'] = (
df_goals_per_game
.groupby(['season', 'goalieId'])['totalGameGoals']
.cumsum()
)


# Group by 'game_id' and 'goalie_id' then sum the 'goal' column to get total goals for each goalie each game
df_shots_on_goal_per_game = df_no_empty_net.groupby(['fullGameId','goalieId','goalieName','season','goalieTeam','isGoalieTeamHome','lastGoalieInNet'])['shotWasOnGoal'].sum().reset_index()

# Rename the column to make it clear that it's the total number of goals
df_shots_on_goal_per_game.rename(columns={'shotWasOnGoal': 'totalShotsOnGoal'}, inplace=True)

# Create a cumulative shots on goal for each goalie in each season
df_shots_on_goal_per_game['cumulativeShotsOnGoal'] = (
df_shots_on_goal_per_game
.groupby(['season', 'goalieId'])['totalShotsOnGoal']
.cumsum()
)


df_merged = pd.merge(df_goals_per_game, df_shots_on_goal_per_game,
                 on=['fullGameId', 'goalieId','goalieName','season','goalieTeam','isGoalieTeamHome','lastGoalieInNet'], how='inner')


df_merged['goalieIdSeasonSavePct'] = ((df_merged['cumulativeShotsOnGoal'] - df_merged['cumulativeGoals']) / df_merged['cumulativeShotsOnGoal'])


# Sort by 'season', 'goalie', and 'game' to ensure order
df_merged = df_merged.sort_values(by=['season', 'goalieId', 'fullGameId']).reset_index(drop=True)

# Create a column for the start of game values for goalie statistics
df_merged['beforeGameSesaonSavePct'] = df_merged.groupby(['season', 'goalieId'])['goalieIdSeasonSavePct'].shift(1).fillna(df_merged['goalieIdSeasonSavePct'].mean()).astype(float)
df_merged['beforeGameSeasonGAA'] = df_merged.groupby(['season', 'goalieId'])['goalieIdSeasonGAA'].shift(1).fillna(df_merged['goalieIdSeasonGAA'].mean()).astype(float)



# Rename fields to match with other data files
df_merged.rename(columns={'fullGameId':'gameId'}, inplace=True)
df_merged.rename(columns={'goalieTeam':'team'}, inplace=True)

# Drop fields used to calc goalie averages
df_merged = df_merged.drop(['totalGameGoals','cumulativeGames','cumulativeGoals','totalShotsOnGoal','cumulativeShotsOnGoal'],axis=1)

df_merged.to_csv(r'Data/goalie_history.csv', index=False)
