# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Nourhan
projectfolder = Path(r"C:\Users\nshafiks\OneDrive - uni-mannheim.de\Documents\GitHub\IS808_adv_DS_lab")

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
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
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data_dir = projectfolder / "data"
similarity_dao_mapping_final_data_set =  data_dir /"similarity_dao_mapping_final_data_set.pq"


df_similarity_dao_mapping_final_data_set = pd.read_parquet(similarity_dao_mapping_final_data_set)
df_similarity_dao_mapping_final_data_set.info()
df_similarity_dao_mapping_final_data_set.describe()


x = df_similarity_dao_mapping_final_data_set['similarity_Art']
y = df_similarity_dao_mapping_final_data_set['nshareddaos_y']

plt.scatter(x, y) 
plt.title('A plot to show the correlation between memes and dankness')
plt.xlabel('Memes')
plt.ylabel('Dankness')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
plt.show()

plt.scatter(x, y) 
plt.show()




# %%
