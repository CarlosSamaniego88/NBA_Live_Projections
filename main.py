from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns
import itertools
import time
import numpy as np
import seaborn as sns
#import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from get_schedule import *
from get_team_info import *
from get_probability import *
from get_home_advantage import *
from get_current_date import *


team_stats = get_team_stats()[0]
opp_stats = get_team_stats()[1]

team_stats = team_stats.sort_values(by = 'Team')
opp_stats = opp_stats.sort_values(by = 'Team')

#print(team_stats)
#print(opp_stats)

#print("Points Allowed:")
points_allowed = list(opp_stats['PTS'])
team_stats['DEF'] = points_allowed
#print(team_stats)
team_stats['MoV'] = team_stats['PTS'] - team_stats['DEF']
team_stats = team_stats.sort_values(by = 'Team')
team_stats['TOV'] = -1 * team_stats['TOV']
print(team_stats)



## making list of all teams
temp_list = list(team_stats['Team'])
list_of_teams = [temp_list[0]]
for team in temp_list:
    if (team != (list_of_teams[len(list_of_teams) - 1])):
        list_of_teams.append(team)
#print("List of teams: " + str(list_of_teams))


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
#print(Y)
#print("\n")
#print(X)


columns = list(X.columns)
#print(columns)
k = (len(columns))
#k = 3    #for testing
R_squared_list, feature_list = [], []
numb_features = []
#print(type(columns))
#print("List of combinations: " + str(list(itertools.combinations(columns, k))))

## Gets R-Squared Value For Each Subset of Variables
i = 1
while (i < k + 1):
    combination_list = list(itertools.combinations(columns, i))
    for combo in combination_list:
        #print(X[list(combo)])
        tmp_result = fit_linear_reg(X[list(combo)], Y)
        R_squared_list.append(tmp_result)
        feature_list.append(combo)
        numb_features.append(len(combo))
    print(i)
    i += 1

## Dataframe of Best Subset From K=1 to K = Total Number of Variables
df = pd.DataFrame({'numb_features': numb_features,'R_squared':R_squared_list,'features':feature_list})
df_max_R_squared = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
print(df_max_R_squared)

## Plots R-Squared Values... Point of Most Curvature if the Number of Features We Should Use
fig = plt.figure(figsize = (16,6))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(df_max_R_squared.numb_features, df_max_R_squared.R_squared, alpha = .2, color = 'darkblue' )
ax.plot(df_max_R_squared.numb_features, df_max_R_squared.R_squared,color = 'r')
ax.set_xlabel('# Features')
ax.set_ylabel('R squared')
ax.set_title('R_squared - Best subset selection')
#ax.legend()

plt.draw()
plt.show()


## Gets The Subset of Features We Want To Use in Our Model
print('\n')
optimal_num_features = input("After looking at plot, how many features do you want to use: ")
optimal_num_features = int(optimal_num_features)

all_subsets = list(df_max_R_squared['features'])
best_subset = list(all_subsets[optimal_num_features - 1])

print('Best subset of features: ' + str(best_subset))


## Make New DataFrame With Only Subset Features
subset_df = team_stats[best_subset]
#print(subset_df)

lin_reg = LinearRegression(fit_intercept = True)
lin_reg = lin_reg.fit(subset_df, Y)

## Formatting Subset_DF To Make Predictions
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
print("Schedule for " + get_todays_date() + ":")
print(todays_schedule_df)

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

print('\n')
print("Margin of Victory Predictions:")
i = 0
mov = 0
while (i < len(visiting_team_projections)):
    mov = abs(round(visiting_team_projections[i][1] - home_team_projections[i][1], 2))
    if (visiting_team_projections[i][1] > home_team_projections[i][1]):
        print(str(visiting_team_projections[i][0]) + " over " + str(home_team_projections[i][0]) + " by " + str(mov))
    else:
        print(str(home_team_projections[i][0]) + " over " + str(visiting_team_projections[i][0]) + " by " + str(mov))
    i += 1

print('\n')
print("Win Percentage Predictions:")
i = 0
mov = 0
while (i < len(visiting_team_projections)):
    mov = abs(round(visiting_team_projections[i][1] - home_team_projections[i][1], 2))
    if (visiting_team_projections[i][1] > (home_team_projections[i][1])):
        away_percentage = get_win_probability(mov)[0]
        home_percentage = get_win_probability(mov)[1]
        print(str(visiting_team_projections[i][0]) + "(" + away_percentage + ") @ " + str(home_team_projections[i][0]) + "(" + home_percentage + ")")
    else:
        home_percentage = get_win_probability(mov)[0]
        away_percentage = get_win_probability(mov)[1]
        print(str(visiting_team_projections[i][0]) + "(" + away_percentage + ") @ " + str(home_team_projections[i][0]) + "(" + home_percentage + ")")
    i += 1




# MCBARLOWE FOR LIVE PLAY BY PLAY INFO
# first run: pip install nba_scraper
# import nba_scraper.nba_scraper as ns
# if you want to return a dataframe
# you can pass the function a list of strings or integers
# all nba game ids have two leading zeros but you can omit these
# to make it easier to create lists of game ids as I add them on
# nba_df = ns.scrape_game([21800001])

# # if you want a csv if you don't pass a file path the default is home
# # directory
# # ns.scrape_game([21800001, 21800002], data_format='csv', data_dir='file/path')
# for col in nba_df.columns:
#     print(col) 

# print("Select specific columns:")
# print(nba_df[['period', 'pctimestring', 'event_type_de', 'score', 'home_team_abbrev', 'away_team_abbrev', 'hs', 'vs']])

# import nba_scraper.nba_scraper as ns

# # if you want to return a dataframe
# # you can pass the function a list of strings or integers
# # all nba game ids have two leading zeros but you can omit these
# # to make it easier to create lists of game ids as I add them on
# nba_df = ns.scrape_game([21800001, 21800002])
# print(nba_df)
