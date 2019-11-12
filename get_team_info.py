from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns
import numpy as np
import requests
from bs4 import BeautifulSoup as soup
from selenium import webdriver

# seasons = ['2016-17', '2017-18', '2018-19', '2019-20']

#https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss
#https://www.basketball-reference.com/leagues/NBA_2019.html#all_team-stats-per_poss
#https://www.basketball-reference.com/leagues/NBA_2018.html#all_team-stats-per_poss
#https://www.basketball-reference.com/leagues/NBA_2017.html#all_team-stats-per_poss

seasons = ['2017', '2018', '2019', '2020']

# d = webdriver.Chrome()
# chromedriver = '/Users/Carlos/Downloads/chromedriver'
# d = webdriver.Chrome(chromedriver)

# for season in seasons:
stats_page = requests.get('https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss')
content = stats_page.content

soup = BeautifulSoup(content, 'html.parser')
table = soup.findAll(name = 'table', attrs= {'id': 'schedule'})

html_str = str(table)

temp_df = pd.read_html(html_str)[0]

    #print(temp_df)
schedule_df = schedule_df.append(temp_df, ignore_index = True, sort=False) 


# schedule_df = schedule_df.drop(['Attend.', 'Notes', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
print(schedule_df)
    # s = soup(d.page_source, 'html.parser').find('table', {'class':'table'})
    # # pd.read_html(str(s))
    # headers, [_, *data] = [i.text for i in s.find_all('th')], [[i.text for i in b.find_all('td')] for b in s.find_all('tr')]

    # final_data = [i for i in data if len(i) > 1]

    # print(final_data)