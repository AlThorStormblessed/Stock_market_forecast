import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, classification_report, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

def num_of_days(day, month, year):
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0
    days += (year - 2012) * 365
    days += sum(months[:month - 1])
    days += day

    return days

print(num_of_days(22, 8, 2023))
        

plt.style.use("dark_background")

stocks = pd.read_csv("BTC-USD.csv")
stocks["Day"] = list(range(1, 4253))
stocks.dropna(inplace = True)

splitted = stocks['Date'].str.split('/', expand=True)
 
stocks['day'] = splitted[0].astype('int')
stocks['month'] = splitted[1].astype('int')
stocks['year'] = splitted[2].astype('int')

stocks.drop(["Ref", "Date"], inplace=True, axis = 1)

print(stocks.head())

stocks["high-open"] = stocks["High"] - stocks["Open"]
stocks["close-high"] = stocks["Close"] - stocks["Low"]
stocks["quarter-end"] = np.where(stocks['month'] % 3 == 0, 1, 0)

print(stocks.groupby("quarter-end").mean())

scaler = StandardScaler()

X, y = stocks[["Day"]], stocks["High"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
lm = LinearRegression()

lm.fit(X_train, y_train)
predict = lm.predict(X_test)

print("Linear regression: ", r2_score(y_test, predict))
print("Percentage error: ", mean_absolute_error(y_test, predict)/np.mean(y_test) * 100)

pickle.dump(lm, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

plt.scatter(X_train["Day"], y_train, s = 0.2)
plt.plot(X_test["Day"], predict, "grey")
plt.ylabel("SENSEX Value")
plt.xlabel("Days since 1 January 2012")
plt.title("Linear Regression on SENSEX")
plt.legend(["True Value", "Predicted Value"])
plt.show()

stocks['target'] = np.where(stocks['Close'].shift(-1) > stocks['Close'], 1, 0)

X, y = stocks[["Day", "high-open", "close-high", "quarter-end"]], stocks["target"]
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2022)
log = LogisticRegression()

log.fit(X_train, y_train)
predict_proba = log.predict_proba(X_test)
predict2 = log.predict(X_test)

print(f"Logistic regression: ", roc_auc_score(y_test, predict_proba[:, 1]))
print(confusion_matrix(y_test, predict2))
print(classification_report(y_test, predict2))