from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns
import itertools
import time
import numpy as np
import seaborn as sns
import glob
#import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from get_schedule import *
from get_team_info import *
from get_probability import *
from get_home_advantage import *
from get_current_date import *
from get_past_team_info import *
from sklearn import tree
from sklearn import ensemble

algo = 1
while (algo < 5):
    year_set = 1
    while (year_set < 5):
        #cv_dataframes = []
        print('Using algo number ' + str(algo))

        ### Getting Team Stats from 2015-2019 into one Dataframe
        team_stats = get_past_team_stats('2015')[0]  
        opp_stats = get_past_team_stats('2015')[1]

        team_stats = team_stats.sort_values(by = 'Team')
        opp_stats = opp_stats.sort_values(by = 'Team')

        team_stats = team_stats.dropna()
        opp_stats = opp_stats.dropna()

        points_allowed = list(opp_stats['PTS'])
        team_stats['DEF'] = points_allowed

        team_stats['MoV'] = team_stats['PTS'] - team_stats['DEF']
        team_stats = team_stats.sort_values(by = 'Team')
        team_stats['TOV'] = -1 * team_stats['TOV']

        team_stats = team_stats.drop(columns = ['PTS', 'DEF', 'G', 'PF', 'MP', 'Rk', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'TRB'], axis = 1)
        team_stats['Team'] = team_stats['Team'].str.replace('*', '')
        #cv_dataframes = cv_dataframes.append(team_stats)

        # making list of all teams
        temp_list = list(team_stats['Team'])
        list_of_teams = [temp_list[0]]
        for team in temp_list:
            if (team != (list_of_teams[len(list_of_teams) - 1])):
                list_of_teams.append(team)

        if (year_set == 1):
            years = []
        elif (year_set == 2):
            years = ['2016']
        elif (year_set == 3):
            years = ['2016', '2017']
        else:
            years = ['2016', '2017', '2018']

        for year in years:

            temp_team_stats = get_past_team_stats(year)[0]
            temp_opp_stats = get_past_team_stats(year)[1]

            temp_team_stats = temp_team_stats.sort_values(by = 'Team')
            temp_opp_stats = temp_opp_stats.sort_values(by = 'Team')

            temp_team_stats = temp_team_stats.dropna()
            temp_opp_stats = temp_opp_stats.dropna()

            points_allowed = list(temp_opp_stats['PTS'])
            temp_team_stats['DEF'] = points_allowed

            temp_team_stats['MoV'] = temp_team_stats['PTS'] - temp_team_stats['DEF']
            temp_team_stats = temp_team_stats.sort_values(by = 'Team')
            temp_team_stats['TOV'] = -1 * temp_team_stats['TOV']

            temp_team_stats = temp_team_stats.drop(columns = ['PTS', 'DEF', 'G', 'PF', 'MP', 'Rk', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'TRB'], axis = 1)
            

            team_stats = team_stats.append(temp_team_stats)
            team_stats['Team'] = team_stats['Team'].str.replace('*', '')
            by_row_index = team_stats.groupby(team_stats.Team)
            team_stats = by_row_index.mean()

            #cv_dataframes = cv_dataframes.append(team_stats)
            #print(team_stats)

        # #     print(year)
        # print('\n')
        # #print(cv_dataframes)
        #print(team_stats)


        ###Getting Matchups and Final Scores of Games from 2015-2019
        months = ['october', 'november', 'december', 'january', 'february', 'march']
        if (year_set == 1):
            years = ['2016']
        elif (year_set == 2):
            years = ['2017']
        elif (year_set == 3):
            years = ['2018']
        else:
            years = ['2019']
        #columns = ['Date', 'Start (ET)', 'Visitor/Neutral', 'PTS', 'Home/Neutral', 'PTS.1']
        schedule_df = pd.DataFrame()

        for year in years:
            for month in months: 
                stats_page = requests.get('https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month))

                content = stats_page.content

                soup = BeautifulSoup(content, 'html.parser')
                table = soup.findAll(name = 'table', attrs= {'id': 'schedule'})

                html_str = str(table)

                temp_df = pd.read_html(html_str)[0]

                schedule_df = schedule_df.append(temp_df, ignore_index = True, sort=False)
            print(year)
        schedule_df = schedule_df.drop(['Attend.', 'Notes', 'Unnamed: 6', 'Unnamed: 7'], axis=1)

        #print(schedule_df['PTS'])
        # print(type(schedule_df['PTS']))

        # print(schedule_df)

        schedule_df['MoV'] = (schedule_df['PTS'].astype('int64') - schedule_df['PTS.1'].astype('int64'))

        #print(schedule_df)


        trueMoV = list(schedule_df['MoV'])

        def fit_linear_reg(X,Y, algo):
            #Fit linear regression model and return R squared values
            if (algo == 1):
                model = LinearRegression(fit_intercept = True)
                model.fit(X,Y)
            elif (algo == 2):
                model = KNeighborsRegressor(n_neighbors=4)
                model.fit(X, Y)
            elif (algo == 3):
                model = tree.DecisionTreeRegressor(criterion='mse')
                model.fit(X, Y)
            else:
                params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
                model = ensemble.GradientBoostingRegressor(**params)
                model.fit(X, Y)
            R_squared = model.score(X,Y)
            return R_squared

        #team_stats = cv_dataframes[3]
        Y = team_stats.dropna().MoV
        if (year_set == 1):
            X = team_stats.dropna().drop(columns = ['MoV', 'Team'], axis = 1)
        else:
            X = team_stats.dropna().drop(columns = ['MoV'], axis = 1)
        columns = list(X.columns)

        k = (len(columns))

        R_squared_list, feature_list = [], []
        numb_features = []

        ## Gets R-Squared Value For Each Subset of Variables
        i = 1
        start_time = time.time()
        while (i < k + 1):
            combination_list = list(itertools.combinations(columns, i))
            for combo in combination_list:  
                tmp_result = fit_linear_reg(X[list(combo)], Y, algo)
                R_squared_list.append(tmp_result)
                feature_list.append(combo)
                numb_features.append(len(combo))
            i += 1

        ## Dataframe of Best Subset From K=1 to K = Total Number of Variables
        df = pd.DataFrame({'numb_features': numb_features,'R_squared':R_squared_list,'features':feature_list})
        df_max_R_squared = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]

        optimal_num_features = 6 #int(optimal_num_features)

        all_subsets = list(df_max_R_squared['features'])
        best_subset = list(all_subsets[optimal_num_features - 1])

        print('Best subset of features: ' + str(best_subset))


        ## Make New DataFrame With Only Subset Features
        subset_df = team_stats[best_subset]
        #print(subset_df)

        if (algo == 1):
            model = LinearRegression(fit_intercept = True)
            model = model.fit(subset_df, Y)
        elif (algo == 2):
            model = KNeighborsRegressor(n_neighbors=4)
            model.fit(subset_df, Y)
        elif (algo == 3):
            model = tree.DecisionTreeRegressor(criterion='mse')
            model = model.fit(subset_df, Y)
        else:
            params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
            model = ensemble.GradientBoostingRegressor(**params)
            model.fit(subset_df, Y)




        # Formatting Subset_DF To Make Predictions
        total_attribute_list = []
        for attribute in best_subset:
            attribute_list = []
            for item in subset_df[attribute]:
                attribute_list.append(item)
            total_attribute_list.append(attribute_list)


        subset_stats_list = []
        i = 0
        while (i < len(list_of_teams)):
            temp_list = []
            for item in total_attribute_list:
                temp_list.append(item[i])
            subset_stats_list.append(temp_list)
            i += 1
        #print(subset_stats_list)

        ## Making Predicitions
        predictions = []
        i = 0
        while (i < len(list_of_teams)):
            predictions.append(round(float(model.predict(np.array([subset_stats_list[i]]))), 3))
            i +=1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Algorithm time taken: " + str(elapsed_time))
        #print(predictions)

        visiting_teams = list(schedule_df['Visitor/Neutral'])
        home_teams = list(schedule_df['Home/Neutral'])
        visiting_team_projections = []
        home_team_projections = []

        for visitor in visiting_teams:
            i = 0
            for team in list_of_teams:
                if visitor == team:
                    visiting_team_projections.append([visitor, predictions[i]])
                i += 1

        home_advantage = round(get_home_advantage(), 2) ## gives advantage to home team
        for host in home_teams:
            i = 0
            for team in list_of_teams:
                if host == team:
                    home_team_projections.append([host, predictions[i] + home_advantage])
                i += 1

        mov_pred = []
        #schedule_df.reset_index(drop = True, inplace=True)
        i = 0
        while (i < len(schedule_df.index)):
            mov = round(visiting_team_projections[i][1] - home_team_projections[i][1], 2)
            mov_pred.append(mov)
            i += 1


        # print(mov_pred)
        # print(i)
        # print(len(mov_pred))
        # print(len(schedule_df.index))

        if (algo == 1):
            filename = 'multi_linear_mse.csv'
        elif (algo == 2):
            filename = 'knn_mse.csv'
        elif (algo == 3):
            filename = 'decision_tree_mse.csv'
        else:
            filename = 'gradient_boosting_mse.csv'
        def append(txt, file):
            with open(file, 'a') as f:
                f.write('\n')
                f.write(txt)


        mse = mean_squared_error(trueMoV, mov_pred)
        append(str(mse), filename)
        #print("MSE for algo number " + str(algo) + " and year_set " + str(year_set) + " = " + str(mse))

        year_set += 1

    algo +=1
