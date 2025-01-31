# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:38:13 2025

@author: ZaneC
"""

import os
import subprocess
import create_shots_history
import nhl_win_classification_preprocessing

import os


if os.path.exists("Data/shots_history.csv"):
    pass
else:
    print("Creating shots_history.csv for initial run")
    create_shots_history.main()

print('Run data pre-processing')
import nhl_win_classification_preprocessing
print('Begin prediction model')
import nhl_win_classification_xgboost_run