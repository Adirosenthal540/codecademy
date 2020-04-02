import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

players_info = pd.read_csv(r"C:\Users\Adi Rosental\Documents\codecademy\ex1\tennis_ace_starting\tennis_ace_starting\tennis_stats.csv")
print (players_info.head())
df = pd.DataFrame(players_info)

x =  df [["FirstServe","FirstServePointsWon","FirstServeReturnPointsWon","SecondServePointsWon","SecondServeReturnPointsWon",
           "Aces","BreakPointsConverted","BreakPointsFaced","BreakPointsOpportunities","BreakPointsSaved","DoubleFaults",\
           "ReturnGamesPlayed","ReturnGamesWon","ReturnPointsWon","ServiceGamesPlayed","ServiceGamesWon","TotalPointsWon",\
           "TotalServicePointsWon"]]

#x =  df [["FirstServe","FirstServePointsWon","FirstServeReturnPointsWon","SecondServePointsWon","SecondServeReturnPointsWon","BreakPointsConverted","BreakPointsFaced","BreakPointsOpportunities","BreakPointsSaved","ReturnGamesWon","ReturnPointsWon","ServiceGamesPlayed","ServiceGamesWon","TotalPointsWon","TotalServicePointsWon"]]

#y = df[["Wins","Losses","Winnings","Ranking"]]
y = df[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
print (len(x_train))
print (len(x_test))
print (len(y_train))
print (len(y_test))
mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_predict = mlr.predict(x_test)
print (mlr.coef_)
print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))
print (y_predict[[]])
plt.scatter(y_test, y_predict)
#plt.plot(range(20000), range(20000))

plt.xlabel("num win: $Y_i$")
plt.ylabel("Predicted win: $\hat{Y}_i$")
# plt.title("Actual Rent vs Predicted Rent")

plt.show()
#coment 1
#add in local
