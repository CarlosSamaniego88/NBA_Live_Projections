from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns

# import nba_scraper.nba_scraper as ns

# BASKETBALL REFERENCE AND BEAUTIFUL SOUP FOR INDIVIDUAL PLAYER STATS
# NBA season we will be analyzing
year = 2019
# URL page we will scraping (see image above)
url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year)
# this is the HTML from the given URL
html = urlopen(url)
soup = BeautifulSoup(html, "lxml")

# use findALL() to get the column headers
soup.findAll('tr', limit=2)
# use getText()to extract the text we need into a list
headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
# exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
headers = headers[1:]
print(headers)

# avoid the first header row
rows = soup.findAll('tr')[1:]
player_stats = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]

stats = pd.DataFrame(player_stats, columns = headers)
print(stats.head(10))






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

import nba_scraper.nba_scraper as ns

# if you want to return a dataframe
# you can pass the function a list of strings or integers
# all nba game ids have two leading zeros but you can omit these
# to make it easier to create lists of game ids as I add them on
nba_df = ns.scrape_game([21800001, 21800002])
print(nba_df)
