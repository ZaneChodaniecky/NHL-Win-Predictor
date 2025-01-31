# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:38:13 2025

@author: ZaneC
"""

import os
import create_shots_history
import nhl_win_classification_preprocessing
import nhl_win_classification_xgboost_run


if os.path.exists("Data/shots_history.csv"):
    pass
else:
    print("Creating shots_history.csv for initial run")
    create_shots_history.main()

print('Run data pre-processing')
nhl_win_classification_preprocessing.main()
print('Begin prediction model')
nhl_win_classification_xgboost_run.main()