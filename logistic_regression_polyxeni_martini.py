# eisagw tis bibliothikes
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# fortwsh dedomenwn
file = "default of credit card clients.xls"
df = pd.read_excel(file, header=1)

# kanw dataset inspection
print("First 5 rows of data:\n", df.head())
print("\nLast 5 rows of data:\n", df.tail())
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Shape (rows, columns):", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nData Info:")
df.info()

#edw ginetai to data cleaning
df.rename(columns={'default payment next month': 'default'}, inplace=True)
df.drop(columns=['ID'], inplace=True)
print("\nMissing values per column:\n", df.isnull().sum())

#diereunhtikh analush dedomenwn
sns.countplot(data=df, x='default')
plt.title('Target Class Distribution')
plt.xlabel('Default (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='default', y='LIMIT_BAL', data=df)
plt.title('Credit Limit vs Default Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['AGE'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['LIMIT_BAL'], kde=True, bins=30)
plt.title('Limit Balance Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='default', y='PAY_AMT1', data=df)
plt.title('PAY_AMT1 vs Default')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='SEX', hue='default', data=df)
plt.title('SEX vs Default')
plt.show()

plt.figure(figsize=(6, 6))
edu_counts = df['EDUCATION'].value_counts()
plt.pie(edu_counts, labels=edu_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Education Distribution')
plt.show()

sns.lmplot(x='LIMIT_BAL', y='BILL_AMT1', data=df, aspect=1.5, height=5)
plt.title('LIMIT_BAL vs BILL_AMT1 with Regression Line')
plt.show()

pair_vars = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
sns.pairplot(df[pair_vars])
plt.show()

#descriptive statistics
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nDescriptive Statistics:\n", df[numeric_vars].describe())

#paradeigma covariance
covariance = df['LIMIT_BAL'].cov(df['BILL_AMT1'])
print(f"\nCovariance between LIMIT_BAL and BILL_AMT1: {covariance:.2f}")

#skewness +kurtosis
print("\nSkewness:\n", df[numeric_vars].skew())
print("\nKurtosis:\n", df[numeric_vars].kurtosis())

#proetoimasia montelou
y = df['default'].values
X_base = df.drop('default', axis=1)

# bazw to clustering feature
y = df['default'].values
X_base = df.drop('default', axis=1)
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_base)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_cluster_scaled)
df['cluster'] = kmeans.labels_
X_full = pd.get_dummies(df.drop('default', axis=1), columns=['cluster'], drop_first=True)

# epilogh me vash to pearson
def select_pearson(X_df, y_arr, thresh=0.2):
    tmp = pd.concat([X_df, pd.Series(y_arr, name='default')], axis=1)
    corr = tmp.corr()['default'].drop('default')
    return X_df.loc[:, corr.abs() >= thresh]

# oi parametroi gia ta senaria
splits = {'70_30': 0.7, '80_20': 0.8}
imbalance_methods = {
    'none': None,
    'smote': SMOTE(random_state=42),
    'under': RandomUnderSampler(random_state=42)
}
selections = {'all': False, 'pearson': True}
scalings = {'scale': True, 'raw': False}

# ta LR montela me vash ton liblinear gia megistes 5000 epanalhpseis
base_lr = LogisticRegression(solver='liblinear', max_iter=5000)
grid_lr = GridSearchCV(
    LogisticRegression(solver='liblinear', max_iter=5000),
    param_grid={'C': [0.1, 1]},
    cv=2,
    n_jobs=1
)
models = {'LR': base_lr, 'LR_grid': grid_lr}

# aksiologhsh twn senariwn, metrhsh tou xronou tous (runtime) kai twn metrikwn tous
results = []
for sp_label, sp_frac in splits.items():
    for imb_label, imb in imbalance_methods.items():
        for sel_label, do_pearson in selections.items():
            for sc_label, do_scale in scalings.items():
                start = time.perf_counter()
                # upoklados gia ta features
                X = X_full.copy()
                if do_pearson:
                    X = select_pearson(X, y)
                # split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X.values, y, train_size=sp_frac, stratify=y, random_state=42
                )
                # diaxeirish imbalance 
                if imb is not None:
                    X_tr, y_tr = imb.fit_resample(X_tr, y_tr)
                # scaling
                if do_scale:
                    scaler = StandardScaler().fit(X_tr)
                    X_tr = scaler.transform(X_tr)
                    X_te = scaler.transform(X_te)
                # h loopa tou montelou
                for m_label, m in models.items():
                    try:
                        m.fit(X_tr, y_tr)
                    except Exception:
                        continue
                    y_pred = m.predict(X_te)
                    y_proba = m.predict_proba(X_te)[:, 1]
                    end = time.perf_counter()
                    runtime = end - start
                    # ypologismos gia metrikes
                    acc = accuracy_score(y_te, y_pred)
                    prec = precision_score(y_te, y_pred, zero_division=0)
                    rec = recall_score(y_te, y_pred, zero_division=0)
                    f1 = f1_score(y_te, y_pred, zero_division=0)
                    auc_s = roc_auc_score(y_te, y_proba)
                    cm = confusion_matrix(y_te, y_pred)
                    # emfanish apotelesmatwn sthn consola
                    print(f"{sp_label}|{imb_label}|{sel_label}|{sc_label}|{m_label} ->")
                    print(f"  Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc_s:.3f}")
                    print("  Confusion Matrix:\n", cm)
                    # katagrafes gia tis metrhseis
                    results.append({
                        'split': sp_label,
                        'imbalance': imb_label,
                        'selection': sel_label,
                        'scaling': sc_label,
                        'model': m_label,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'auc': auc_s,
                        'runtime': runtime
                    })
                    
# bazw ta apotelesmata se ena dataframe
res_df = pd.DataFrame(results)

# Optikopoihsh
sns.set(style='whitegrid')

# heatmaps gia Accuracy kai AUC
pivot = res_df.pivot_table(
    index=['split','imbalance','selection','scaling'],
    columns='model',
    values=['accuracy','auc']
)
for metric in ['accuracy','auc']:
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot[metric], annot=True, fmt='.2f', cmap='viridis')
    plt.title(f"{metric.capitalize()} across Scenarios")
    plt.tight_layout()
    plt.show()
    
# barplot AUC ana montelo kai split
plt.figure(figsize=(8,5))
sns.barplot(data=res_df, x='model', y='auc', hue='split', palette='tab10')
plt.title("AUC by Model & Split")
plt.tight_layout()
plt.show()

# AUC ana methodo imbalance 
sns.catplot(
    data=res_df, x='imbalance', y='auc', hue='model', col='split',
    kind='bar', palette='tab10', height=4, aspect=0.9
)
plt.subplots_adjust(top=0.85)
plt.suptitle('AUC by Imbalance Method & Split')
plt.show()

# facetgrid: AUC se selection kai scaling
g = sns.FacetGrid(res_df, row='selection', col='scaling', margin_titles=True)
g.map_dataframe(
    sns.barplot, x='imbalance', y='auc', hue='model', order=['none','smote','under'], palette='tab10'
)
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('AUC by Imbalance across Selection & Scaling')
plt.show()

# ROC curves gia kathe split (xwris imbalance)
for sp_label in splits:
    subset = res_df[(res_df['split']==sp_label) & (res_df['imbalance']=='none')]
    plt.figure(figsize=(8,6))
    for _, row in subset.iterrows():
        model = base_lr if row['model']=='LR' else grid_lr
        X = X_full.values
        if row['selection']=='pearson':
            X = select_pearson(pd.DataFrame(X_full), y).values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=splits[sp_label], stratify=y, random_state=42
        )
        y_proba = model.fit(X_tr, y_tr).predict_proba(X_te)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        auc_val = roc_auc_score(y_te, y_proba)
        plt.plot(fpr, tpr, label=f"{row['model']} (AUC={auc_val:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"ROC Curves: {sp_label} split, none")
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout(); plt.show()

# confusion matrix heatmaps gia ta prwta 4 senaria ana split
for sp_label in splits:
    first4 = res_df[res_df['split']==sp_label].head(4)
    plt.figure(figsize=(10,8))
    for i, row in enumerate(first4.itertuples()):
        ax = plt.subplot(2,2,i+1)
        X = X_full.values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=splits[sp_label], stratify=y, random_state=42
        )
        y_pred = base_lr.fit(X_tr, y_tr).predict(X_te)
        cm = confusion_matrix(y_te, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(f"{row.model} | {sp_label} | {row.imbalance}")
        ax.set_xlabel('Pred'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
# to pio grhgoro senario genikws
fast = res_df.loc[res_df['runtime'].idxmin()]
print("\nFastest scenario overall:", fast.to_dict())

# optikopoihsh gia to pio grhgoro senario
X = X_full.values
if fast.selection=='pearson':
    X = select_pearson(pd.DataFrame(X_full), y).values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=splits[fast.split], stratify=y, random_state=42
)
if fast.imbalance!='none':
    imb = imbalance_methods[fast.imbalance]
    X_tr, y_tr = imb.fit_resample(X_tr, y_tr)
if fast.scaling=='scale':
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
model = base_lr if fast.model=='LR' else grid_lr

# ROC gia to pio grhgoro senario
y_proba = model.fit(X_tr, y_tr).predict_proba(X_te)[:,1]
fpr, tpr, _ = roc_curve(y_te, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_te,y_proba):.2f}")
plt.plot([0,1],[0,1],'k--')
plt.title('Fastest Scenario ROC')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()
# confusion matrix gia to pio grhgoro senario
y_pred = model.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Fastest Scenario Confusion')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()