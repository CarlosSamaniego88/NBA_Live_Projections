import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

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

    #print(temp_df)
    schedule_df = schedule_df.append(temp_df, ignore_index = True, sort=False)

## returns slate of games for entire regular season
schedule_df = schedule_df.drop(['Attend.', 'Notes', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
print(schedule_df.head(50))

## returns slate of games for a specific date
print("\n")
day_schedule_df = schedule_df[schedule_df['Date'] == 'Wed, Oct 23, 2019']
print(day_schedule_df)