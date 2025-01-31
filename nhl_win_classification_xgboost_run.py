
# coding: utf-8

import pandas as pd
import numpy as np
import os
import datetime
import json
import requests
import time
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from datetime import datetime, timedelta, timezone



def main(): 
    
        # Check available GPUs
        gpu_count = get_gpu_device_count()
        if gpu_count > 0:
            print(f"GPU is available with {gpu_count} GPU(s).")
        else:
            print("No GPU detected. Ensure your setup is correct.")

        # Test CatBoost with GPU
        try:
            model = CatBoostClassifier(task_type="GPU", devices='0')  # Specify GPU
            device = 'cuda'
            print("CatBoost can use the GPU.")
                
        except Exception as e:
            device = 'cpu'
            print(f"Error: {e}")
            

        # Declare important variables
        train_path = 'Game_Win_Classifier_Transformed.csv'
        test_path = 'Game_Win_Classifier_Predict.csv'
        target_variable = 'win_or_lose_Home'
        SEED = 69
        skip_hypertuning = True


        def load_dataframes(path1: str, path2: str, format1: str = "csv", format2: str = "csv") -> tuple:

            loaders = {
                "csv": pd.read_csv,
                "excel": pd.read_excel,
                "json": pd.read_json,
            }

            if format1 not in loaders or format2 not in loaders:
                raise ValueError("Unsupported format. Supported formats: 'csv', 'excel', 'json'.")

            # Load the dataframes using appropriate loaders
            df1 = loaders[format1](path1)
            df2 = loaders[format2](path2)

            print(f"Loading data from {os.getcwd()}")
            
            return df1, df2


        df_train, df_test = load_dataframes(train_path,test_path)


        df_train = df_train.drop(columns=['team_Home','team_Away','gameId','home_or_away_Home','home_or_away_Away','win_or_lose_Away'])
        df_test = df_test.drop(columns=['goalieId_Home','goalieName_Home','goalieId_Away','goalieName_Away'])


        df_train = df_train.rename(columns={target_variable: 'y'})


        def get_num_cols(df: pd.DataFrame):
            num_cols = df.select_dtypes(include=['number']).columns.to_list()
            
            return num_cols

        def get_cat_cols(df: pd.DataFrame):
            cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()  

            return cat_cols


        def anova_test(data, cat_cols, target):
            anova_res = []
            for col in cat_cols:
                groups = [data[target][data[col] == cat] for cat in data[col].unique()]
                f_stat, p_val = f_oneway(*groups)
                anova_res.append([col, f_stat, p_val])
            return anova_res


        # Sort predictive categorical columns
        cat_cols = get_cat_cols(df_train.drop('y', axis=1))

        anova_res = anova_test(df_train, cat_cols, 'y')
        anova_df = pd.DataFrame(anova_res, columns = ['Column', 'F Statistic', 'P-value'])
        anova_df = anova_df.sort_values(ascending = False, by = 'P-value')


        # Sort predictive numerical columns
        num_cols = get_num_cols(df_train)

        corr_col = df_train[num_cols].corr()['y'].drop(['y'])
        corr_df = corr_col.reset_index()
        corr_df.columns = ['Columns', 'Correlation']
        corr_df = corr_df.sort_values(ascending = False, by = 'Correlation').reset_index(drop=True)


        # Select best numerical and categorical colums 
        sel_cat = np.array(anova_df[anova_df['P-value'] < 0.03]['Column'])
        sel_cat = get_cat_cols(df_train.drop('y', axis=1))

        sel_num = np.array(corr_df[np.abs(corr_df['Correlation']) > 0.03]['Columns'])
        sel_num = get_num_cols(df_train.drop('y', axis=1))


        # Split home and away features
        top_home_columns = [col for col in df_train[sel_num].columns if col.endswith('_Home')]
        top_away_columns = [col for col in df_train[sel_num].columns if col.endswith('_Away')]


        # Filters out high VIF Home features
        # Standardize features
        X = df_train[top_home_columns]  # Excludes the target column
        X_scaled = StandardScaler().fit_transform(X)

        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

        # Sort the DataFrame by 'VIF' in ascending order
        vif_data = vif_data.sort_values(by='VIF')

        # Filter the DataFrame to include only the rows where VIF is below the threshold
        low_vif_home_features = vif_data[vif_data['VIF'] <= 10]

        # Display the selected features
        print(low_vif_home_features)


        # Filters out high VIF Away features
        # Standardize features
        X = df_train[top_away_columns]  # Excludes the target column
        X_scaled = StandardScaler().fit_transform(X)

        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

        # Sort the DataFrame by 'VIF' in ascending order
        vif_data = vif_data.sort_values(by='VIF')

        # Filter the DataFrame to include only the rows where VIF is below the threshold
        low_vif_away_features = vif_data[vif_data['VIF'] <= 10]

        # Display the selected features
        print(low_vif_away_features)


        best_cols = low_vif_home_features['Feature'].to_list() + low_vif_away_features['Feature'].to_list()


        # Split into 80% training data and 20% testing data
        X = df_train[best_cols].copy()
        y = df_train['y'].copy()

        feature_names = X.columns.tolist()

        # Define test size and validation size relative to the training data
        test_size = 0.2
        val_size = 0.2  # Validation size relative to the training data

        # Create train/test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)

        # Create train/val split from the training set (80% of training data for training, 20% for validation)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=SEED)

        # Check the sizes of each set (for verification)
        print(f"Training set size: x = {X_train.shape[0]} | y = {y_train.shape[0]}")
        print(f"Validation set size: x = {X_val.shape[0]} | y = {y_val.shape[0]}")
        print(f"Test set size: x = {X_test.shape[0]} | y = {y_test.shape[0]}")


        # Pipeline constructor used to run transformation steps in order
        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])

        X_train_prepared= num_pipeline.fit_transform(X_train)

        X_val_prepared= num_pipeline.transform(X_val)
        X_test_prepared= num_pipeline.transform(X_test)




        # Hyperparameters for BayesSearchCV tuning
        search_space = {
            'n_estimators': (50, 500),                   # Number of trees
            'max_depth': (3, 15),                        # Depth of each tree
            'learning_rate': (0.01, 0.1, 'log-uniform'), # Learning rate
            'subsample': (0.5, 0.9),                     # Fraction of samples for training
            'colsample_bytree': (0.5, 0.9),              # Fraction of features per tree
            'gamma': (0, 5),                             # Minimum loss reduction
            'reg_alpha': (0, 10),                        # L1 regularization (sparsity)
            'reg_lambda': (1, 10),                       # L2 regularization (penalize large coefficients)
            'min_child_weight': (1, 20)                  # Tuning min_child_weight within the range
        }

        EARLY_STOPPING = 10
        ITERATIONS = 50
        FOLDS = 5

        print(f'Tuning hyperparameters ({ITERATIONS} iterations with {FOLDS} folds)...')
        # Leave some cores available
        available_cores = os.cpu_count()
        n_cores = max(1, available_cores - 2)  # Leave 3 cores free

        model = XGBClassifier(random_state=SEED, early_stopping_rounds=EARLY_STOPPING, verbosity=0)

        # Set up BayesSearchCV for hyperparameter tuning
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=ITERATIONS,          # Number of iterations for optimization
            cv=FOLDS,               # k-fold cross-validation
            verbose=0,          # Display detailed logs
            scoring="neg_log_loss",
            random_state=SEED,
            n_jobs=n_cores      # Use available cores
        )

        if skip_hypertuning:
            best_params = {"colsample_bytree": 0.9, "gamma": 3, "learning_rate": 0.1, "max_depth": 3,
                           "min_child_weight": 20, "n_estimators": 500, "reg_alpha": 1, "reg_lambda": 1,
                           "subsample": 0.9}
        else:
            start_time = time.time()
            
            bayes_search.fit(X_train_prepared, y_train,
                             eval_set=[(X_val_prepared, y_val)])
            
            end_time = time.time()  
            
            best_params = bayes_search.best_params_
            
            print("Best parameters:", best_params)

            # Save best parameters to a file
            with open("best_params.json", "w") as f:
                json.dump(best_params, f)

            # Calculate elapsed time in minutes
            elapsed_time_minutes = (end_time - start_time) / 60
            print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")



        # Initialize the XGBRegressor
        model = XGBClassifier(
            **best_params,
            early_stopping_rounds = EARLY_STOPPING,
            device='cpu',
            random_state=SEED        # Ensure reproducibility
        )

        # Train the model
        model.fit(
            X_train_prepared, 
            y_train, 
            eval_set=[(X_val_prepared, y_val)],
            verbose=0
        )


        # Accuracy is how correct overall predictions are
        y_train_pred = model.predict(X_train_prepared)
        print(f"Train Accuracy = {round(accuracy_score(y_train, y_train_pred), 5)}")

        y_test_pred = model.predict(X_test_prepared)
        print(f"Test Accuracy = {round(accuracy_score(y_test, y_test_pred), 5)}") 


        # How many of the win predicitons were correct?
        print(f"Train Precision = {round(precision_score(y_train, y_train_pred), 5)}")
        print(f"Test Precision = {round(precision_score(y_test, y_test_pred), 5)}") 


        # How many of the actual wins did you predict?
        print(f"Train Recall = {round(recall_score(y_train, y_train_pred), 5)}")
        print(f"Test Recall = {round(recall_score(y_test, y_test_pred), 5)}") 


        # Prepare inference data
        X = df_test[best_cols].copy()
        X_test_prepared= num_pipeline.transform(X)


        # Prepare list of results
        results_dict = {1 : 'win', 0 : 'lose'}

        home_team_results = list(model.predict(X_test_prepared))

        for i, item in enumerate(home_team_results):
            if item in results_dict:
                home_team_results[i] = results_dict[item]


        # Create dataframe to map team names and nightly odds
        df_results = pd.DataFrame(columns=['Home','Home_Name','Home_Odds','Home_Result',
                                   'Away','Away_Name','Away_Odds'])


        # Load team names
        df_acronyms = pd.read_csv('Data/team_abbreviations.csv', encoding="ISO-8859-1")  
        # Convert DataFrame to dictionary
        teams_dict = df_acronyms.set_index('Short_Name')['Long_Name'].to_dict()


        away_teams = df_test.iloc[:,0].tolist()
        home_teams = df_test.iloc[:,1].tolist()


        df_results['Home'] = home_teams
        df_results['Home_Name'] = df_results['Home'].map(teams_dict)
        df_results['Home_Result'] = home_team_results
        df_results['Away'] = away_teams
        df_results['Away_Name'] = df_results['Away'].map(teams_dict)
        df_results['Away_Result'] = df_results['Home_Result'].apply(lambda x: 'lose' if x == 'win' else 'win')


        # Create datetime to only pull todays games
        current_date = datetime.today().strftime('%Y-%m-%d')
        #commence_time_to = '2024-09-27T00:15:00Z'

        current_datetime = datetime.now(timezone.utc)
        # Set limit as tomorrow at 6am to include any late eastern time games
        tomorrow_datetime = current_datetime + timedelta(days=1)
        custmo_tomorrow_datetime = tomorrow_datetime.replace(hour=6,minute=0,second=0,microsecond=0)

        # Convert to ISO 8601 format
        commence_time_to = custmo_tomorrow_datetime.isoformat().replace('+00:00','Z')


        # Pull game odds from odds-api
        odds_URL = 'https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?'\
                   'apiKey=94588626e22d896e9f196d0745f00928&bookmakers=draftkings&markets=h2h&'\
                   f'oddsFormat=american&commenceTimeTo={commence_time_to}'


        # API is limited to 500 pulls per month so only use when needed
        result = requests.get(odds_URL)

        # Check if the request was successful
        if result.status_code == 200:
            # Parse the JSON response
            data = result.json()
            
            teams_of_interest = ['Dallas Cowboys','New York Giants']
            odds_list = []
            teams_list = []
            # Extracting 'price' values from the JSON data
            for game in data:
                bookmakers = game.get('bookmakers', [])
                if bookmakers:
                    markets = bookmakers[0].get('markets', [])
                    if markets:
                        outcomes = markets[0].get('outcomes', [])
                        if outcomes:
                            for outcome in outcomes:
                                team_name = outcome.get('name', 'N/A')
                                if team_name not in teams_of_interest:  # Check if team is in the list
                                    price = outcome.get('price', 'N/A')
                                    odds_list.append(price)
                                    teams_list.append(team_name)
                                    #print(f'Team: {team_name}, Price: {price}')
        else:
            print(f"Failed to retrieve data. Status code: {result.status_code}")
            

        # Create dataframe with games, predictions, and odds
        odds_dict = dict(zip(teams_list,odds_list))

    df_results['Home_Odds'] = df_results['Home_Name'].map(odds_dict)
    df_results['Away_Odds'] = df_results['Away_Name'].map(odds_dict)

    df_results.to_csv('game_predictions.csv', index=False)
    df_results.head(df_results.shape[0])


# Prevent code from running when the module is imported
if __name__ == "__main__":
    main()