# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Nourhan
projectfolder = Path(r"C:\Users\nshafiks\OneDrive - uni-mannheim.de\Documents\GitHub\IS808_adv_DS_lab")

# %%
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
# import the metrics class
from sklearn import metrics
# import required modules
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# %%
data_dir = projectfolder / "data"
voter_dao_similarity_mapping =  data_dir /"voter_dao_similarity_mapping.pq"


df_voter_dao_similarity_mapping = pd.read_parquet(voter_dao_similarity_mapping)
df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'] = df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'].fillna(0)
df_voter_dao_similarity_mapping.info()

# %% Linear Regression
x_array = df_voter_dao_similarity_mapping[["cosine_similarity based on count vector node2vec"]].to_numpy()
y_array = df_voter_dao_similarity_mapping[["nshareddaos_x"]].to_numpy()

model = LinearRegression().fit(x_array, y_array)
r_sq = model.score(x_array, y_array)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# %% Logistic Regression
feature_cols = ['cosine_similarity based on count vector node2vec', 'similarity_Art', 'similarity_Collectible', 'similarity_Games','similarity_Other','similarity_Utility']
X = df_voter_dao_similarity_mapping[feature_cols] # Independent Variables
y = df_voter_dao_similarity_mapping.nshareddaos_x # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)
# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

# %%
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
target_names = ['Do not vote in the same DAO', 'vote in the same DAO']
print(classification_report(y_test, y_pred, target_names=target_names))

# %%
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# %%
# x = np.array(['cosine_similarity based on count vector node2vec']).reshape((-1, 1))
# y = np.array(['nshareddaos'])

# model = LinearRegression().fit(x, y)
# r_sq = model.score(x, y)

#model = LogisticRegression(solver='liblinear', random_state=0)
#model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
#confusion_matrix(y, model.predict(x))
#print(f"coefficients: {model.coef_}")