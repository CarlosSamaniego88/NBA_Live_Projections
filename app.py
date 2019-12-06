from flask import Flask, render_template
import glob
from get_schedule import *
from get_team_info import *
from get_probability import *
from get_home_advantage import *
from get_current_date import *
from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4 import Comment
import pandas as pd
import nba_scraper.nba_scraper as ns
import itertools
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import date
import requests
import re
import xlsxwriter

#WHOLE SCRIPT ORCHESTRATED BY CARLOS SAMANIEGO
#CODE WRITTEN BY ALEX, ANDREW

app = Flask(__name__)

@app.route('/')
def display_predictions():
    team_stats = get_team_stats(2020)[0]
    opp_stats = get_team_stats(2020)[1]

    team_stats = team_stats.sort_values(by = 'Team')
    opp_stats = opp_stats.sort_values(by = 'Team')

    points_allowed = list(opp_stats['PTS'])
    team_stats['DEF'] = points_allowed

    team_stats['MoV'] = team_stats['PTS'] - team_stats['DEF']
    team_stats = team_stats.sort_values(by = 'Team')
    team_stats['TOV'] = -1 * team_stats['TOV']


    # making list of all teams
    temp_list = list(team_stats['Team'])
    list_of_teams = [temp_list[0]]
    for team in temp_list:
        if (team != (list_of_teams[len(list_of_teams) - 1])):
            list_of_teams.append(team)

    ##########################Choosing The Optimal Model###############################

    ## Linear Regression Function
    def fit_linear_reg(X,Y):
        #Fit linear regression model and return R squared values
        model_k = LinearRegression(fit_intercept = True)
        model_k.fit(X,Y)
        R_squared = model_k.score(X,Y)
        return R_squared

    ## Split Up Response Variable From Predictors and Drop Unnecessary Columns
    Y = team_stats.dropna().MoV
    X = team_stats.dropna().drop(columns = ['PTS', 'MoV', 'DEF', 'Team', 'G', 'PF', 'MP', 'Rk', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'TRB'], axis = 1)

    columns = list(X.columns)

    k = (len(columns))

    R_squared_list, feature_list = [], []
    numb_features = []

    ## Gets R-Squared Value For Each Subset of Variables
    i = 1
    while (i < k + 1):
        combination_list = list(itertools.combinations(columns, i))
        for combo in combination_list:  
            tmp_result = fit_linear_reg(X[list(combo)], Y)
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

    ## Make New DataFrame With Only Subset Features
    subset_df = team_stats[best_subset]

    lin_reg = LinearRegression(fit_intercept = True)
    lin_reg = lin_reg.fit(subset_df, Y)

    # Formatting Subset_DF To Make Predictions
    total_attribute_list = []
    for attribute in best_subset:
        attribute_list = []
        for item in subset_df[attribute]:
            attribute_list.append(item)
        total_attribute_list.append(attribute_list)
    #print(total_attribute_list)

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
        predictions.append(round(float(lin_reg.predict(np.array([subset_stats_list[i]]))), 3))
        i +=1

    #print(predictions)

    ## Getting Schedule
    todays_schedule_df = get_todays_games(get_todays_date())
    todays_schedule_df = todays_schedule_df.drop(columns = ['Date'], axis = 1)

    ## Assigning Score Predictions to Each Matchup
    visiting_teams = list(todays_schedule_df['Visitor/Neutral'])
    home_teams = list(todays_schedule_df['Home/Neutral'])
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

    i = 0

    list_of_teams.sort()
    nba_logos = glob.glob('static/*.png')
    nba_logos.sort()

    projec_d = {}
    mov = 0
    while (i < len(visiting_team_projections)):
        mov = abs(round(visiting_team_projections[i][1] - home_team_projections[i][1], 2))
        if (visiting_team_projections[i][1] > (home_team_projections[i][1])):
            away_percentage = get_win_probability(mov)[0]
            home_percentage = get_win_probability(mov)[1]
            if "static/"+str(visiting_team_projections[i][0])+".png" in nba_logos:
                projec_d["static/"+str(visiting_team_projections[i][0])+".png"] = away_percentage
            if "static/"+str(home_team_projections[i][0])+".png" in nba_logos:
                projec_d["static/"+str(home_team_projections[i][0])+".png"] = home_percentage
        else:
            home_percentage = get_win_probability(mov)[0]
            away_percentage = get_win_probability(mov)[1]
            if "static/"+str(visiting_team_projections[i][0])+".png" in nba_logos:
                projec_d["static/"+str(visiting_team_projections[i][0])+".png"] = away_percentage
            if "static/"+str(home_team_projections[i][0])+".png" in nba_logos:
                projec_d["static/"+str(home_team_projections[i][0])+".png"] = home_percentage
        i+=1
    #LINE 180 is Carlos Samaniego, Alex Brockman
    #HOME.HTML made by Carlos Samaniego and Alex Brockman
    return render_template('home.html', len = len(projec_d), projec_d=projec_d, k = list(projec_d.keys()), v = list(projec_d.values()))

if __name__=="__main__":
    app.run(debug=True)