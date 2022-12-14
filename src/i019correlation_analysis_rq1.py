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
import matplotlib.pyplot as plt



# %%
data_dir = projectfolder / "data"
similarity_dao_mapping_final_data_set =  data_dir /"similarity_dao_mapping_final_data_set.pq"


df_similarity_dao_mapping_final_data_set = pd.read_parquet(similarity_dao_mapping_final_data_set)

# %%

# x = df_similarity_dao_mapping_final_data_set['cosine_similarity']
# y = df_similarity_dao_mapping_final_data_set ['nshareddaos_y']

# time = [0, 1, 2, 3]
# position = [0, 100, 200, 300]

# plt.plot(x, y)
# plt.xlabel('Time (hr)')
# plt.ylabel('Position (km)')

import seaborn as sns
sns.scatterplot(x="similarity_total", y="nshareddaos_y", data=df_similarity_dao_mapping_final_data_set);

# plt.plot(x, y)
# plt.show()

# plt.scatter(x, y)
# plt.show()

# %%
