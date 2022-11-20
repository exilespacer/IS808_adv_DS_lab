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
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import randn
from numpy.random import seed
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns


# %%
data_dir = projectfolder / "data"
voter_dao_similarity_mapping =  data_dir /"voter_dao_similarity_mapping.pq"


df_voter_dao_similarity_mapping = pd.read_parquet(voter_dao_similarity_mapping)
df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'] = df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'].fillna(0)
df_voter_dao_similarity_mapping.info()

# %% Correlation Analysis
# calculate Pearson's correlation

print('pearson correlation: %.3f' % df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'].corr(df_voter_dao_similarity_mapping['nshareddaos_y'], method='pearson'))
print('Spearman correlation: %.3f' % df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'].corr(df_voter_dao_similarity_mapping['nshareddaos_y'], method='spearman'))
print('kendall correlation: %.3f' % df_voter_dao_similarity_mapping['cosine_similarity based on count vector node2vec'].corr(df_voter_dao_similarity_mapping['nshareddaos_y'], method='kendall'))
df_voter_dao_similarity_mapping.corr()

# %%
sns.heatmap(df_voter_dao_similarity_mapping.corr(), vmin=-1, vmax=1,
annot=True,cmap="rocket_r")
plt.show()