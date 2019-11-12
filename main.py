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
# print(stats.head(10))
# print('\n')
# print(len(stats.columns))

##########################Choosing The Optimal Model###############################

## Linear Regression Function
def fit_linear_reg(X,Y):
    #Fit linear regression model and return R squared values
    model_k = LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    R_squared = model_k.score(X,Y)
    return R_squared

## Split Up Response Variable From Predictors and Drop Unnecessary Columns
Y = stats.dropna().PTS
X = stats.dropna().drop(columns = ['PTS', 'Player', 'Pos', 'Tm', 'G', 'GS', 'PF', 'Age', 'FG%', '3P%', '2P%', 'FT%', 'eFG%', 'MP'], axis = 1)
#print(Y)
#print("\n")
#print(X)


columns = list(X.columns)
#print(columns)
k = (len(columns))
#k = 5    #for testing
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

plt.show()







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
