import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


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