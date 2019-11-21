from urllib.request import urlopen, Request
import pandas as pd
import nba_scraper.nba_scraper as ns
import numpy as np
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import re

#In this file we retreive the 'Team Per 100 Possessions Stats' for use

# Original link: https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss
def get_team_stats():
    seasons = ['2017', '2018', '2019', '2020']

    #NBA season
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