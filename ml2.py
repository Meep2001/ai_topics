import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = np.array([
    [800], [1000], [1200], [1400],
    [1600], [1800], [2000], [2200],
    [2400], [2600]
])

y = np.array([
    250000, 300000, 350000, 400000,
    450000, 500000, 550000, 600000,
    650000, 700000
])

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(Y_test, predictions)
print('Predictions:', predictions)
print('Actual:', Y_test)
print('MAE:', mae)

new_house = np.array([[1500]])
price = model.predict(new_house)
print('Predicted price:',price)