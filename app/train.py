import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("../Data/hotel_booking_cancellation.csv")

# Convert arrival_date_month to numerical
months = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']
df['arrival_date_month'] = df['arrival_date_month'].apply(lambda x: months.index(x) + 1)

# One-hot encode categorical columns
one_hot_cols = ['hotel','meal','market_segment','distribution_channel',
                'deposit_type','customer_type','continent']
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)


# Add engineered features
df['high_risk_flag'] = ((df['previous_cancellations'] > 0) & (df['lead_time'] > 100)).astype(int)
df['cancel_ratio'] = df['previous_cancellations'] / (df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1)


X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Lighter model — fewer trees, shallower depth, less RAM usage
#rf_tuned = RandomForestClassifier(
    #n_estimators=50,          # reduced from 250 → much less memory
    #criterion='gini',
    #max_depth=12,             # reduced from 24 → smaller trees
    #min_samples_split=10,     # increased → fewer splits, smaller trees
    #min_samples_leaf=5,       # increased → avoids tiny leaves
    #max_features='sqrt',      # only use sqrt(n_features) per split
    #class_weight={0: 1, 1: 2},
    #random_state=0,
    #n_jobs=-1
#)

rf_tuned = RandomForestClassifier(
    n_estimators=100,     # up from 50
    max_depth=16,         # up from 12
    min_samples_split=6,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight={0: 1, 1: 2},
    random_state=0,
    n_jobs=-1
)

rf_tuned.fit(X_train, y_train)

y_train_pred = rf_tuned.predict(X_train)
y_test_pred  = rf_tuned.predict(X_test)

print("==== Train Data Evaluation ====")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

print("\n==== Test Data Evaluation ====")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

joblib.dump(rf_tuned, "../model/model.pkl")
print("\n[INFO] Model saved to model/model.pkl")
print("Feature columns:", X.columns.tolist())