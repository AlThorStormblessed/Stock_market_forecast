# Stock_market_forecast

This project creates a linear regression model for India's SENSEX and lets users make predictions for its value on different dates through a web application. Partway through adding a logistic regression model that will predict whether the market will close higher or lower than the current day by using the current day's open, close, high and low values.

## Model

The linear regression model uses SENSEX data of the previous 12 years (since January 1 2012) to train and test, then predict values of "High" for any given date. The R-squared value for the prediction comes out to be around 0.9, showing a very good linear relationship, and has a percentage error of roughly 9% during testing. 
The logistic regression model uses the same data, and makes use of the various parameters of each day to train and test the data and predict whether the next day will see the market go up or down given a set of these parameters. This currently has an accuracy of around 55%.

## Web Application

This project uses Flask to create a simple web application for users to input a date and get a rough prediction for the value of SENSEX on that day. 

## Running the project

1. Download Stock.py, app.py, request.py and BTC-USD.csv into a folder, Stock.html as ./templates/Stock.html and style.css as ./static/css/style.css
2. Run Stock.py normally to create a .pkl model file.
3. Run app.py in terminal as 'python app.py' and open the local link to run the web application.

## Future

Firstly the web application will be expanded to include the logistic regression model. Later on, better ML techniques will be used to improve upon the prediction models and increase accuracy, and the web application will be improved in terms of usability and aesthetics as well.
