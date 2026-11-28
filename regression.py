import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = {
    "size": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "bedrooms": [2, 2, 3, 3, 4, 4, 4, 5],
    "price": [250000, 300000, 360000, 450000,
              540000, 600000, 660000, 750000]
}

df = pd.DataFrame(data)
X = df[["size","bedrooms"]]
y = df["price"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

print("Intercept:", model.intercept_)
print("Coef:", model.coef_)
predictions = model.predict(X_test)
results = pd.DataFrame({
    "Actual Price": Y_test,
    "Predicted": predictions
})
print(results)
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
r2_score = r2_score(Y_test, predictions)
print('MAE:',mae)
print('MSE:', mse)
print('R2 score:', r2_score)

