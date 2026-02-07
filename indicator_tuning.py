import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

df = pd.read_csv('combined_stock_data.csv')
# todo: drop cols with NaN; Adjusted close vs close
# print(df.loc[0])

# If I buy at close today → will price rise ≥2% in 5 days?
df['future_max_5d'] = (
    df.groupby('Ticker')['High']
    .shift(-1)
    .rolling(10)
    .max()
)
df['target'] = ((df['future_max_5d'] / df['Close'] - 1) > 0.02).astype(int)
df = df.dropna()

# feature matrix
features = [
    'EMA20', 'EMA50', 'MACD Line', 'Signal Line', 'RSI_14', 'RSI_4',
    'IV', 'WILLR', '%K', '%D', 'Volume', 'Open', 'High', 'Low', 'Close'
]
X = df[features]
y = df['target']

# split data -> before 2024 is used to train (starting mid 2020); after is test
SPLIT_DATE = '2024-01-01'
train = df['Date'] < SPLIT_DATE
test = df['Date'] >= SPLIT_DATE
X_train, y_train = X[train], y[train]
X_test,  y_test = X[test],  y[test]

# train xgboost
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=8
)

model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))

df_test = df[test].copy()
df_test['proba'] = proba
trades = df_test[df_test['proba'] > 0.7]
print('Number of trades:', len(trades))
print('Win rate:', trades['target'].mean())

# find which indicators matter
imp = pd.Series(model.feature_importances_, index=features)
print(imp.sort_values(ascending=False))

# RSI(4) + WillR + IV is the real signal

#               precision    recall  f1-score   support

#            0       0.80      0.80      0.80     87547
#            1       0.90      0.91      0.90    181187

#     accuracy                           0.87    268734
#    macro avg       0.85      0.85      0.85    268734
# weighted avg       0.87      0.87      0.87    268734

# Number of trades: 161344
# Win rate: 0.9434996033320111
# RSI_4          0.510936
# WILLR          0.198841
# IV             0.117079
# %D             0.033955
# RSI_14         0.021758
# %K             0.019250
# MACD Line      0.014822
# Signal Line    0.014178
# Volume         0.012426
# High           0.010577
# Low            0.010240
# EMA50          0.009497
# EMA20          0.009323
# Close          0.009095
# Open           0.008022
