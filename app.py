from flask import Flask, render_template
import glob
# from get_schedule import *
# from get_team_info import *
# from get_probability import *
# from get_home_advantage import *
# from get_current_date import *
# from main import *             #fake main
from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4 import Comment
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
from datetime import date
import requests
import re
import xlsxwriter

app = Flask(__name__)

@app.route('/')
def display_predictions():
    # teams = {'static/Miami Heat.png': '53.5%', 'static/Milwaukee Bucks.png':'25.5%'}
    team_stats = get_team_stats()[0]
    opp_stats = get_team_stats()[1]

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

    ## Plots R-Squared Values... Point of Most Curvature if the Number of Features We Should Use
    # fig = plt.figure(figsize = (16,6))
    # ax = fig.add_subplot(1, 2, 1)
    # ax.scatter(df_max_R_squared.numb_features, df_max_R_squared.R_squared, alpha = .2, color = 'darkblue' )
    # ax.plot(df_max_R_squared.numb_features, df_max_R_squared.R_squared,color = 'r')
    # ax.set_xlabel('# Features')
    # ax.set_ylabel('R squared')
    # ax.set_title('R_squared - Best subset selection')
    # #ax.legend()

    # plt.draw()
    # plt.show()


    ## Gets The Subset of Features We Want To Use in Our Model
    #optimal_num_features = input("After looking at plot, how many features do you want to use: ")
    optimal_num_features = 6 #int(optimal_num_features)

    all_subsets = list(df_max_R_squared['features'])
    best_subset = list(all_subsets[optimal_num_features - 1])

    # print('Best subset of features: ' + str(best_subset))


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
    # print("Schedule for " + get_todays_date() + ":")
    # print(todays_schedule_df)

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

    #Spread Predictions
    # print('\n')
    # print("Margin of Victory Predictions:")
    # i = 0
    # mov = 0
    # while (i < len(visiting_team_projections)):
    #     mov = abs(round(visiting_team_projections[i][1] - home_team_projections[i][1], 2))
    #     if (visiting_team_projections[i][1] > home_team_projections[i][1]):
    #         # print(str(visiting_team_projections[i][0]) + " over " + str(home_team_projections[i][0]) + " by " + str(mov))
    #     else:
    #         # print(str(home_team_projections[i][0]) + " over " + str(visiting_team_projections[i][0]) + " by " + str(mov))
    #     i += 1

    #Win Probabilities
    # print('\n')
    # print("Win Percentage Predictions:")
    i = 0
    # fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
    # axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]
    # fig1.suptitle('Win Probabilities for ' + get_todays_date())
    # plt.rcParams['font.size'] = 7.0

    # mov = 0
    # while (i < len(visiting_team_projections)):
    #     mov = abs(round(visiting_team_projections[i][1] - home_team_projections[i][1], 2))
    #     if (visiting_team_projections[i][1] > (home_team_projections[i][1])):
    #         away_percentage = get_win_probability(mov)[0]
    #         home_percentage = get_win_probability(mov)[1]
    #         print(str(visiting_team_projections[i][0]) + "(" + away_percentage + ") @ " + str(home_team_projections[i][0]) + "(" + home_percentage + ")")

    #     else:
    #         home_percentage = get_win_probability(mov)[0]
    #         away_percentage = get_win_probability(mov)[1]
    #         print(str(visiting_team_projections[i][0]) + "(" + away_percentage + ") @ " + str(home_team_projections[i][0]) + "(" + home_percentage + ")")
        
    #     labels = visiting_team_projections[i][0], home_team_projections[i][0]
    #     sizes = [float(away_percentage.strip('%')), float(home_percentage.strip('%'))]
    #     explode = (0.1, 0)
    #     #fig1, ax1 = plt.subplots()
    #     axes[i].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #     shadow=True, startangle=90)
    #     axes[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    #     plt.draw()
    #     i += 1

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
    # print(projec_d)

    # while (i < len(axes)):
    #     axes[i] = axes[i].set_visible(False)
    #     i += 1
    # plt.show()

    # MCBARLOWE FOR LIVE PLAY BY PLAY INFO
    # first run: pip install nba_scraper
    # import nba_scraper.nba_scraper as ns
    # if you want to return a dataframe
    # you can pass the function a list of strings or integers
    # all nba game ids have two leading zeros but you can omit these
    # to make it easier to create lists of game ids as I add them on
    # nba_df = ns.scrape_game([21800001])

    # if you want a csv if you don't pass a file path the default is home
    # directory
    # ns.scrape_game([21800001, 21800002], data_format='csv', data_dir='file/path')
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
    # j = 0
    return render_template('home.html', len = len(projec_d), projec_d=projec_d, k = list(projec_d.keys()), v = list(projec_d.values()))

#In this file we retreive the 'Team Per 100 Possessions Stats' for use
# Original link: https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss
def get_team_stats():
    seasons = ['2017', '2018', '2019', '2020']

    year = 2020

    url = 'https://www.basketball-reference.com/leagues/NBA_{}.html'.format(year)
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    #remove comments so we can access the correct table
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables = []
    for each in comments:
        if 'table' in each:
            try:
                tables.append(pd.read_html(each)[0])
            except:
                continue

    team_stats = tables[4]
    opp_stats = tables[5]
    return team_stats, opp_stats

def get_todays_games(todays_date):
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
    #columns = ['Date', 'Start (ET)', 'Visitor/Neutral', 'PTS', 'Home/Neutral', 'PTS.1']
    schedule_df = pd.DataFrame()

    for month in months: 
        stats_page = requests.get('https://www.basketball-reference.com/leagues/NBA_2020_games-{}.html'.format(month))

        content = stats_page.content

        soup = BeautifulSoup(content, 'html.parser')
        table = soup.findAll(name = 'table', attrs= {'id': 'schedule'})

        html_str = str(table)

        temp_df = pd.read_html(html_str)[0]

        schedule_df = schedule_df.append(temp_df, ignore_index = True, sort=False)

    ## returns slate of games for entire regular season
    schedule_df = schedule_df.drop(['Attend.', 'Notes', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
    #print(schedule_df.head(50))

    ## making list of all dates
    temp_list = list(schedule_df['Date'])
    list_of_dates = [temp_list[0]]
    for date in schedule_df['Date']:
        if (date != (list_of_dates[len(list_of_dates) - 1])):
            list_of_dates.append(date)

    ## returns slate of games for a specific date
    print("\n")
    day_schedule_df = schedule_df[schedule_df['Date'] == todays_date]
    return day_schedule_df


def get_todays_date():
    # Getting Calendar Date
    today = date.today()
    date1 = today.strftime("%b %d, %Y")

    # Getting weekday
    weekdays = ['Mon, ', 'Tue, ', 'Wed, ', 'Thu, ', 'Fri, ', 'Sat, ', 'Sun, ']
    weekday = weekdays[today.weekday()]

    #Combining Weekday and Calendar Date
    todays_date = weekday + date1
    #print(todays_date)
    new_date = ''
    for letter in todays_date[:10]:
        if letter != '0':
            new_date += letter
    new_date += todays_date[10:]
    # print(new_date)

    return new_date

def get_home_advantage():
    pace = requests.get('https://www.teamrankings.com/nba/stat/average-scoring-margin', "html.parser")
    soup = BeautifulSoup(pace.content, "lxml")
    rows = soup.find_all('tr')
    str_cells = str(rows)

    list_rows = []
    for row in rows:
        cells = row.find_all('td')
        str_cells = str(cells)
        clean = re.compile('<.*?>')
        clean2 = (re.sub(clean, '',str_cells))
        list_rows.append(clean2)

    df = pd.DataFrame(list_rows)

    df1 = df[0].str.split(', ', expand=True)

    df1[0] = df1[0].str.strip('[')
    df1[0] = df1[0].str.strip(']')
    df1[5] = df1[5].str.strip('+')

    df1 = df1.drop(df1.index[0])
    home_margins_list = list(df1[5])

    sum = 0
    for margin in home_margins_list:
        sum += float(margin)

    home_average_margin = sum / 30

    return home_average_margin

def get_win_probability(projectedSpread):

    probabilities = pd.read_csv("spread_to_probability.csv")
    favorite_percentage = ''
    underdog_percentage = ''

    spreads = list(probabilities['Spread'])
    favorite_percentages = list(probabilities['Home'])
    underdog_percentages = list(probabilities['Away'])

    i = 0
    for i in range(len(spreads)):
        if ((projectedSpread >= spreads[i]) and (projectedSpread <= spreads[i + 1])):
            favorite_percentage = favorite_percentages[i]
            underdog_percentage = underdog_percentages[i]
        
    return favorite_percentage, underdog_percentage

if __name__=="__main__":
    app.run(debug=True)