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
from sklearn.metrics import mean_squared_error
from get_schedule import *
from get_team_info import *
from get_probability import *
from get_home_advantage import *
from get_current_date import *


### Getting Team Stats from 2015-2019 into one Dataframe
team_stats = get_team_stats('2015')[0]  
opp_stats = get_team_stats('2015')[1]

team_stats = team_stats.sort_values(by = 'Team')
opp_stats = opp_stats.sort_values(by = 'Team')

team_stats = team_stats.dropna()
opp_stats = opp_stats.dropna()

points_allowed = list(opp_stats['PTS'])
team_stats['DEF'] = points_allowed

team_stats['MoV'] = team_stats['PTS'] - team_stats['DEF']
team_stats = team_stats.sort_values(by = 'Team')
team_stats['TOV'] = -1 * team_stats['TOV']

years = ['2016', '2017', '2018', '2019']

for year in years:

    temp_team_stats = get_team_stats(year)[0]
    temp_opp_stats = get_team_stats(year)[1]

    temp_team_stats = temp_team_stats.sort_values(by = 'Team')
    temp_opp_stats = temp_opp_stats.sort_values(by = 'Team')

    temp_team_stats = temp_team_stats.dropna()
    temp_opp_stats = temp_opp_stats.dropna()

    points_allowed = list(temp_opp_stats['PTS'])
    temp_team_stats['DEF'] = points_allowed

    temp_team_stats['MoV'] = temp_team_stats['PTS'] - temp_team_stats['DEF']
    temp_team_stats = temp_team_stats.sort_values(by = 'Team')
    temp_team_stats['TOV'] = -1 * temp_team_stats['TOV']

    team_stats = team_stats.append(temp_team_stats)

    print(year)

#print(team_stats)


###Getting Matchups and Final Scores of Games from 2015-2019
months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
years = ['2015', '2016', '2017', '2018', '2019']
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

print(schedule_df)
