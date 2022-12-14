# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Nourhan
projectfolder = Path(r"C:\Users\nshafiks\OneDrive - uni-mannheim.de\Documents\GitHub\IS808_adv_DS_lab")

# %%
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn import datasets, linear_model
from sklearn import linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

#%%
data_dir = projectfolder / "data"
coparticipation_similarity_mapping =  data_dir /"coparticipation_similarity_mapping.pq"

df_coparticipation_similarity_mapping = pd.read_parquet(coparticipation_similarity_mapping)

#%%Multiple Logistic Regression
X=df_coparticipation_similarity_mapping['similarity_category_distance'].values.reshape(-1, 1)
y=df_coparticipation_similarity_mapping['share_daos'].values

#Source
#https://blog.finxter.com/logistic-regression-scikit-learn-vs-statsmodels/   

X = sm.add_constant(X)
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#%% Linear Regression
features = 'similarity_category_distance'
target = 'number_shared_daos'

X = df_coparticipation_similarity_mapping[features].values.reshape(-1, 1)
y = df_coparticipation_similarity_mapping[target].values

#statsmodel
Xx = df_coparticipation_similarity_mapping[['similarity_category_distance']].values.reshape(-1, 1) 
yy = df_coparticipation_similarity_mapping['number_shared_daos'].values 
## fit a OLS model with intercept on TV and Radio 
Xx = sm.add_constant(Xx) 
est = sm.OLS(yy, Xx)
result=est.fit()
print(result.summary())