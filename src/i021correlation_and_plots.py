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

#df_coparticipation_similarity_mapping.count()
df_new = pd.DataFrame(df_coparticipation_similarity_mapping.loc[df_coparticipation_similarity_mapping['similarity_total'] == 1])
df_new.count()

#%% DONE Comembership Stats
df_coparticipation_similarity_mapping['share_daos'].value_counts()
sns.countplot(x='share_daos', data=df_coparticipation_similarity_mapping, palette='hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(df_coparticipation_similarity_mapping[df_coparticipation_similarity_mapping['share_daos']==0])
count_sub = len(df_coparticipation_similarity_mapping[df_coparticipation_similarity_mapping['share_daos']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no participation is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of participation", pct_of_sub*100)

#%% DONE Comembership by Category
table=pd.crosstab(df_coparticipation_similarity_mapping.similarity_total,df_coparticipation_similarity_mapping.share_daos)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, subplots=True)
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#plt.title('Co-participation by total categories')
plt.xlabel('Total categories')
plt.ylabel('Percentage')
plt.savefig('Co-participation_by_total_categories')
plt.show()

#%% DONE Covoting Stats
df_covotor_similarity_mapping['share_choices'].value_counts()
sns.countplot(x='share_choices', data=df_covotor_similarity_mapping, palette='hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(df_covotor_similarity_mapping[df_covotor_similarity_mapping['share_choices']==0])
count_sub = len(df_covotor_similarity_mapping[df_covotor_similarity_mapping['share_choices']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no covoting is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of covoting", pct_of_sub*100)

#%%
ax = df_coparticipation_similarity_mapping.plot.hexbin(x='similarity_category_distance', y='number_shared_daos', gridsize=15)

#%%
x = df_coparticipation_similarity_mapping['similarity_category_distance']
y = df_coparticipation_similarity_mapping['number_shared_daos']
df = pd.DataFrame(x, y)
# plot it
sns.heatmap(df)
#%% DONE Covoting by Category
table=pd.crosstab(df_covotor_similarity_mapping.similarity_total,df_covotor_similarity_mapping.share_choices)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, subplots=True)
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

#plt.title('Co-voting by total categories')
plt.xlabel('Total categories')
plt.ylabel('Percentage')
plt.savefig('Co-voting_by_total_categories')
plt.show()
#%%
sns.scatterplot(x="similarity_category_distance", y="number_shared_daos", data=df_coparticipation_similarity_mapping);
#%%
sns.scatterplot(x="similarity_nft_distance", y="number_shared_daos", data=df_coparticipation_similarity_mapping);
#%%
sns.scatterplot(x="pct_similar1st_avg", y="number_shared_daos", data=df_coparticipation_similarity_mapping);
#%%
sns.scatterplot(x="pct_similar2nd_avg", y="number_shared_daos", data=df_coparticipation_similarity_mapping);
#%%
sns.scatterplot(x="similarity_category_distance", y="number_shared_choices", data=df_covotor_similarity_mapping);
#%%
sns.scatterplot(x="similarity_nft_distance", y="number_shared_choices", data=df_covotor_similarity_mapping);
#%%
sns.scatterplot(x="pct_similar2nd_avg", y="number_shared_choices", data=df_covotor_similarity_mapping);
# plt.plot(x, y)
# plt.show()

# plt.scatter(x, y)
# plt.show()


#%%
# df_similarity_dao_mapping_final_data_set['share_daos'].value_counts()
# sns.countplot(x='share_daos', data=df_similarity_dao_mapping_final_data_set, palette='hls')
# plt.show()
# plt.savefig('count_plot')

# count_no_sub = len(df_similarity_dao_mapping_final_data_set[df_similarity_dao_mapping_final_data_set['share_daos']==0])
# count_sub = len(df_similarity_dao_mapping_final_data_set[df_similarity_dao_mapping_final_data_set['share_daos']==1])
# pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
# print("percentage of no subscription is", pct_of_no_sub*100)
# pct_of_sub = count_sub/(count_no_sub+count_sub)
# print("percentage of subscription", pct_of_sub*100)

# #%matplotlib inline
# pd.crosstab(df_similarity_dao_mapping_final_data_set.similarity_total,df_similarity_dao_mapping_final_data_set.share_daos).plot(kind='bar')
# plt.title('Similarity')
# plt.xlabel('Similarity')
# plt.ylabel('Count')
# plt.savefig('similarities_count')
# plt.show()

# table=pd.crosstab(df_similarity_dao_mapping_final_data_set.similarity_total,df_similarity_dao_mapping_final_data_set.share_daos)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Similarity')
# plt.xlabel('Similarity')
# plt.ylabel('Count')
# plt.savefig('similarities_count')
# plt.show()

# df_similarity_dao_mapping_final_data_set[["similarity_Utility", "number_shared_daos"]].corr()
# plt.figure(figsize=(5, 5))
# ax = plt.axes()
# ax.scatter(df_similarity_dao_mapping_final_data_set["similarity_Utility"], df_similarity_dao_mapping_final_data_set["number_shared_daos"], color='b', alpha=0.20)
# ax.set_xlabel('similarity_Utility')
# ax.set_ylabel('number_shared_daos')
# plt.show()
# plt.savefig('img/utility_dao_scatter.png')