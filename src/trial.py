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
import matplotlib.pyplot as plt
# import the metrics class
from sklearn import metrics
# import required modules
import seaborn as sns
from numpy.random import randn
from numpy.random import seed
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn import linear_model as lm
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %% Final Data sets
data_dir = projectfolder / "data"

coparticipation_similarity_mapping = data_dir /"coparticipation_similarity_mapping.pq"
covotor_similarity_mapping = data_dir /"covotor_similarity_mapping.pq"

df_coparticipation_similarity_mapping = pd.read_parquet(coparticipation_similarity_mapping)
df_covotor_similarity_mapping = pd.read_parquet(covotor_similarity_mapping)

#%%
ax = df_coparticipation_similarity_mapping.plot.hexbin(x='similarity_category_distance', y='number_shared_daos', gridsize= 20)
#%%
##ax = df_covotor_similarity_mapping.plot.hexbin(x='similarity_category_distance', y='number_shared_choices', gridsize=10)

# %%
print("Correlation between Similarity category distance and number of shared DAOs")
print(df_coparticipation_similarity_mapping['number_shared_daos'].corr(df_coparticipation_similarity_mapping['similarity_category_distance']))
print("Correlation between Similarity category distance and number of shared choices")
print(df_covotor_similarity_mapping['similarity_category_distance'].corr(df_covotor_similarity_mapping['number_shared_choices']))

print(df_covotor_similarity_mapping['pct_similar1st_avg'].corr(df_covotor_similarity_mapping['number_shared_choices']))
print(df_covotor_similarity_mapping['pct_similar2nd_avg'].corr(df_covotor_similarity_mapping['number_shared_choices']))
print(df_covotor_similarity_mapping['similarity_nft_distance'].corr(df_covotor_similarity_mapping['number_shared_choices']))

print(df_covotor_similarity_mapping['similarity_total'].corr(df_covotor_similarity_mapping['number_shared_choices']))
print(df_coparticipation_similarity_mapping['numeric_owned_nft_kinds_x'].corr(df_coparticipation_similarity_mapping['number_shared_daos']))
print(df_coparticipation_similarity_mapping['numeric_owned_nft_kinds_y'].corr(df_coparticipation_similarity_mapping['number_shared_daos']))

print(df_coparticipation_similarity_mapping['similarity_category_distance'].corr(df_coparticipation_similarity_mapping['share_daos']))
print(df_coparticipation_similarity_mapping['numeric_owned_nft_kinds_x'].corr(df_coparticipation_similarity_mapping['share_daos']))
#%%
# fig = plt.figure(figsize=(15,10))
# sns.heatmap(df.corr(method='spearman'), annot = True, cmap="Blues")
# plt.title("Correlation Heatmap")
