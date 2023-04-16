import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('data/train.csv',header = None)

X = df_train.iloc[:,1:-1]  
Y = df_train.iloc[:, -1]  

le = LabelEncoder()

# Fit and transform the 'city' column
X['city_encoded'] = le.fit_transform(X[1])
X.drop(1, axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.rename(columns={2:"Second", 3: "Third", 4:"Fourth"})
y_train.name = 'Label'


X_val = X_val.rename(columns={2:"Second", 3: "Third", 4:"Fourth"})
y_val.name = 'Label'

model = LogisticRegression()
model.fit(X_train,y_train)
print("model trained")
#predicted_labels = model.predict(X_val)
#accuracy = accuracy_score(y_val, predicted_labels)

#unique_values, value_counts = np.unique(predicted_labels, return_counts=True)
#class_counts = dict(zip(unique_values, value_counts))
#print('Class counts:', class_counts)

with open('artifacts/label_enc.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save trained model
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)