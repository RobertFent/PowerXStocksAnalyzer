import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

df = pd.read_csv('combined_stock_data.csv')
# print(df.loc[0])

# If I buy at close today → will price rise ≥2% in 5 days?
df['future_max_5d'] = (
    df.groupby('Ticker')['High']
    .shift(-1)
    .rolling(5)
    .max()
)
df['target'] = ((df['future_max_5d'] / df['Close'] - 1) > 0.02).astype(int)
df = df.dropna()

# feature matrix
features = [
    'EMA20', 'EMA50', 'MACD Line', 'Signal Line', 'RSI_14', 'RSI_4',
    'IV', 'WILLR_4', 'WILLR_14', '%K', '%D', 'Volume', 'Open', 'High', 'Low', 'Close'
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

# WILLR(4) + RSI(4) + IV is the real signal

#               precision    recall  f1-score   support

#            0       0.81      0.85      0.83    121963
#            1       0.87      0.83      0.85    146735

#     accuracy                           0.84    268698
#    macro avg       0.84      0.84      0.84    268698
# weighted avg       0.84      0.84      0.84    268698

# Number of trades: 118978
# Win rate: 0.9262384642538957
# WILLR_4        0.373412
# RSI_4          0.226372
# IV             0.184858
# WILLR_14       0.062423
# %D             0.031194
# %K             0.019694
# RSI_14         0.015891
# Volume         0.015313
# High           0.011454
# Signal Line    0.011139
# MACD Line      0.010624
# Low            0.009981
# Close          0.007748
# EMA50          0.007543
# EMA20          0.006370
# Open           0.005984
