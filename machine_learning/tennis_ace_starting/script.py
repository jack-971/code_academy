import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
#print(df.head())
#print(len(df))

# perform exploratory analysis here:
plt.scatter(df['Year'], df['Winnings'])
plt.xlabel("Year")
plt.ylabel("Winnings")
plt.title("Year vs Winnings")
plt.show()
plt.clf()

plt.scatter(df['BreakPointsOpportunities'], df['Winnings'])
plt.xlabel("Break Point Opportunities")
plt.ylabel("Winnings")
plt.title("Break Point Opportunities vs Winnings")
plt.show()
plt.clf()

plt.scatter(df['BreakPointsFaced'], df['FirstServe'])
plt.xlabel("Break Points Faced")
plt.ylabel("First Serves In")
plt.title("BreakPoints Faced vs First Serves in")
plt.show()
plt.clf()

## perform single feature linear regressions here:
x_train, x_test, y_train, y_test = train_test_split(df[['BreakPointsOpportunities']], df[['Winnings']], test_size=0.2)
#x_train, x_test, y_train, y_test = train_test_split(df[['BreakPointsSaved']], df[['ServiceGamesWon']], test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predicted = model.predict(x_test)


plt.scatter(y_test, y_predicted, alpha=0.4)
plt.xlabel("Test Data")
plt.ylabel("Predicted Data")
plt.title("Predicted Vs Test Data")
plt.show()
plt.clf()

## perform two feature linear regressions here:
x_train, x_test, y_train, y_test = train_test_split(df[['BreakPointsOpportunities','BreakPointsSaved']], df[['Winnings']], test_size=0.2)
#x_train, x_test, y_train, y_test = train_test_split(df[['BreakPointsSaved']], df[['ServiceGamesWon']], test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predicted = model.predict(x_test)


plt.scatter(y_test, y_predicted, alpha=0.4)
plt.xlabel("Test Data")
plt.ylabel("Predicted Data")
plt.title("Predicted Vs Test Data")
plt.show()
plt.clf()

## perform multiple feature linear regressions here:
x_train, x_test, y_train, y_test = train_test_split(df[['BreakPointsOpportunities','BreakPointsSaved', 'Aces']], df[['Winnings']], test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predicted = model.predict(x_test)


plt.scatter(y_test, y_predicted, alpha=0.4)
plt.xlabel("Test Data")
plt.ylabel("Predicted Data")
plt.title("Predicted Vs Test Data")
plt.show()
