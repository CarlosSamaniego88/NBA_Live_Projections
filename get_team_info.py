from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns
import numpy as np
import requests
from bs4 import BeautifulSoup as soup
from selenium import webdriver

seasons = ['2016-17', '2017-18', '2018-19', '2019-20']

# d = webdriver.Chrome()
chromedriver = '/Users/Carlos/Downloads/chromedriver'
d = webdriver.Chrome(chromedriver)
season = '2016-17'
# d.get('https://stats.nba.com/teams/advanced/?sort=W&dir=-1&Season={}&SeasonType=Regular%20Season'.format(season))
for season in seasons:
    d.get('https://stats.nba.com/teams/advanced/?sort=W&dir=-1&Season={}&SeasonType=Regular%20Season'.format(season))

    s = soup(d.page_source, 'html.parser').find('table', {'class':'table'})

    headers, [_, *data] = [i.text for i in s.find_all('th')], [[i.text for i in b.find_all('td')] for b in s.find_all('tr')]

    final_data = [i for i in data if len(i) > 1]

    print(final_data)

    
# BASKETBALL REFERENCE AND BEAUTIFUL SOUP FOR INDIVIDUAL PLAYER STATS
# NBA season we will be analyzing
# for season in seasons:
#     stats_page = requests.get('https://stats.nba.com/teams/advanced/?sort=W&dir=-1&Season={}&SeasonType=Regular%20Season'.format(season))

#     content = stats_page.content

#     soup = BeautifulSoup(content, 'html.parser')
#     table = soup.findAll(name = 'table', attrs= {'id': 'schedule'})

#     html_str = str(table)

#     temp_df = pd.read_html(html_str)[0]

#     #print(temp_df)
#     schedule_df = schedule_df.append(temp_df, ignore_index = True, sort=False) 


# schedule_df = schedule_df.drop(['Attend.', 'Notes', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
# print(schedule_df)