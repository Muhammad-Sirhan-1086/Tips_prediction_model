import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data_set= pd.read_csv("tips_dataset.csv")
data_set_encoded = pd.get_dummies(data_set, drop_first=True) # converting string type data
# manipulating data before split to avoid any miss match later
# Define features and target
X = data_set_encoded.drop("tip", axis=1)
y = data_set_encoded["tip"]
model_sc = LinearRegression()
X_top = X[['total_bill', 'size']]
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y, test_size=0.2, random_state=42)
model_sc= LinearRegression().fit(X_train_top, y_train_top)
y_predict_top = model_sc.predict(X_test_top)
#print("MSE:", mean_squared_error(y_test_top, y_predict_top))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), X[['total_bill', 'size']], y, scoring='neg_mean_squared_error', cv=5)
#print("cross validation mean squared error:", -scores.mean())
#print("Mean tip:", y.mean())
#print("Standard deviation of tips:", y.std())
mse = mean_squared_error(y_test_top, y_predict_top,)

import numpy as np
rmse = np.sqrt(mse)
#print("rmse value",rmse)

# Input feature names used during training
feature_names = ['total_bill', 'size']

while True:
    try:
        # Take input
        x = list(map(float, input("Enter 'total_bill' and 'size' separated by space: ").split()))
        if len(x) != 2:
            print("❌ Please enter exactly two values.")
            continue

        # Format input for prediction
        x_df = pd.DataFrame([x], columns=feature_names)
        prediction = model_sc.predict(x_df)
        print(f"💡 Predicted Tip: {prediction[0]:.2f}")

        # Ask user if they want to continue
        choice = input("Would you like to make another prediction? (yes/no): ").strip().lower()
        if choice not in ['yes', 'y']:
            print("👋 Okay, goodbye! Have a great day!")
            break

    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")


