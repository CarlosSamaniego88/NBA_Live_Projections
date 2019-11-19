from flask import Flask, render_template
import glob
from get_schedule import *
from get_team_info import *
from fake_main import *             #fake main
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

app = Flask(__name__)

#need to output schedule of games, our predictions
# @app.route('/schedule')
# def display_team_schedule():
#     return todays_schedule_df.to_html(header='true', table_id='table')

    #print("List of teams: " + str(list_of_teams))
    # return render_template('home.html', tables=[team_stats.to_html(classes='data')], titles=team_stats.columns.values)
    # return render_template('home.html',  team_stats.to_html(header='true', table_id='table'))
    # return render_template("home.html", data=team_stats.to_html())
    # return team_stats.to_html(header='true', table_id='table')        #works right now
    # return todays_schedule_df.to_html(header='true', table_id='table')

@app.route('/')
def display_predictions():
    # i = 0
    # predictions = []
    # while (i < len(visiting_team_projections)):
    #     if (visiting_team_projections[i][1] > home_team_projections[i][1]):
    #         predictions.append(str(visiting_team_projections[i][0]) + " over " + str(home_team_projections[i][0]) + " by " + str(round(visiting_team_projections[i][1] - home_team_projections[i][1], 2)))
    #     else:
    #         predictions.append(str(home_team_projections[i][0]) + " over " + str(visiting_team_projections[i][0]) + " by " + str(round(home_team_projections[i][1] - visiting_team_projections[i][1], 2)))
    #     i += 1

    # {% for team, photo in zipped %}
    #     <h3> {{team}}, {{photo}} </h3>
    # {% endfor %}

    return '<img src="/Users/Carlos/Projects/NBA_Live_Projections/images/atlanta.png"/>'
    # return render_template('home.html', zipped=zipped)


if __name__=="__main__":
    app.run(debug=True)