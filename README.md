# NBA_Live_Projections


To get what is currently on the website please following the following instructions:
Before running:
"export FLASK_APP=app.py" and "export FLASK_DEBUG=1"

To Run:
"flask run"

to stop --> Control C

Site should be located at "http://localhost:5000/schedule" or "http://localhost:5000/predictions

To get more info on how our model works, please run main.py.

If you run the main.py script you should get a dataframe table of team stats along with a set of number of features with an associated graph that will tell you the best subset selection. Once you choose the number of features, it will display the predicted margins of victory based on the number of features.

Main.py and the website (app.py) have not been completely integrated yet.

Disclaimer, the fake main is just a tool to test with outputting information on the website. There is no point in running fake_main.py by itseld
