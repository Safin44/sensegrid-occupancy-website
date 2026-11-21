"""Replicates Lab_Project__232_35_189.ipynb exactly (same features, split, pipelines),
verifies metrics against the notebook's recorded outputs, exports the best model."""
import json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

CSV = '/home/hatch/workspace/user/files/pirvision_office_dataset1.csv'
df = pd.read_csv(CSV)
data = df.copy()
dt = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='coerce')
data['Hour'] = dt.dt.hour; data['Minute'] = dt.dt.minute; data['Second'] = dt.dt.second; data['DayOfWeek'] = dt.dt.dayofweek
data = data.drop(columns=['Date', 'Time'])
X = data.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
y = data['Label']
print("X shape:", X.shape, "| columns:", list(X.columns)[:3], "...", list(X.columns)[-4:])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("train/test:", X_train.shape, X_test.shape)

models = {
    'Decision Tree': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', DecisionTreeClassifier(random_state=42, class_weight='balanced'))]),
    'Random Forest': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))]),
    'KNN': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=5))]),
    'SVM': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', SVC(kernel='rbf', class_weight='balanced'))]),
}
expected = {  # notebook recorded outputs (cell 11)
    'Random Forest': (0.994121, 0.988832, 0.987224, 0.987955),
    'KNN': (0.990856, 0.979027, 0.985891, 0.982431),
    'Decision Tree': (0.986283, 0.975506, 0.977107, 0.976303),
    'SVM': (0.983018, 0.949900, 0.993067, 0.970151),
}
rows, trained = [], {}
for name, m in models.items():
    m.fit(X_train, y_train); trained[name] = m
    p = m.predict(X_test)
    r = dict(model=name,
             accuracy=round(accuracy_score(y_test, p), 6),
             precision=round(precision_score(y_test, p, average='macro', zero_division=0), 6),
             recall=round(recall_score(y_test, p, average='macro', zero_division=0), 6),
             f1=round(f1_score(y_test, p, average='macro', zero_division=0), 6))
    rows.append(r)
    e = expected[name]
    ok = all(abs(r[k] - v) < 5e-5 for k, v in zip(['accuracy','precision','recall','f1'], e))
    print(f"{name}: acc={r['accuracy']} prec={r['precision']} rec={r['recall']} f1={r['f1']} | notebook match: {ok}")

rf = trained['Random Forest']
yp = rf.predict(X_test)
cm = confusion_matrix(y_test, yp, labels=[0,1,3]).tolist()
print("RF confusion matrix [0,1,3]:", cm)
rep = classification_report(y_test, yp, labels=[0,1,3], target_names=['Vacancy','Stationary Human','Other Activity'], output_dict=True, zero_division=0)
imp = pd.DataFrame({'Feature': X.columns, 'Importance': rf.named_steps['model'].feature_importances_}).sort_values('Importance', ascending=False)
print("Top 8 importances:\n", imp.head(8).to_string(index=False))

# first test sample check (notebook cell 18: actual 0, predicted 0)
s0 = X_test.iloc[[0]]; print("first test sample: actual", y_test.iloc[0], "predicted", rf.predict(s0)[0])

# presets: first correctly-predicted test row of each class
preds = pd.Series(yp, index=y_test.index)
presets = {}
for cls, key in [(0,'vacancy'), (1,'stationary'), (3,'activity')]:
    idx = [i for i in y_test.index if y_test[i]==cls and preds[i]==cls][0]
    presets[key] = {'actual_label': int(cls), 'features': {c: float(X_test.loc[idx, c]) for c in X.columns}}
    print("preset", key, "row idx", idx)

joblib.dump(rf, '/home/hatch/workspace/occupancy_3d_website/backend/smart_room_occupancy_model.pkl')
metrics = {
  'models': sorted(rows, key=lambda r: -r['accuracy']),
  'rf_confusion_matrix': cm, 'rf_report': {k: rep[k] for k in ['Vacancy','Stationary Human','Other Activity']},
  'rf_accuracy': rep['accuracy'],
  'top_features': [{'feature': f, 'importance': round(float(v), 6)} for f, v in zip(imp['Feature'].head(15), imp['Importance'].head(15))],
  'feature_order': list(X.columns),
  'class_distribution': {str(int(k)): int(v) for k, v in y.value_counts().items()},
  'first_test_sample': {'actual': int(y_test.iloc[0]), 'predicted': int(rf.predict(s0)[0])},
}
json.dump(metrics, open('/home/hatch/workspace/occupancy_3d_website/backend/metrics.json','w'), indent=1)
json.dump(presets, open('/home/hatch/workspace/occupancy_3d_website/backend/presets.json','w'), indent=1)
print("saved: smart_room_occupancy_model.pkl, metrics.json, presets.json")
