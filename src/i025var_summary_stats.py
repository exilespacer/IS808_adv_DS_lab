# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
# projectfolder = Path("/project/IS808_adv_DS_lab")
# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")
# Mengnan 

projectfolder = Path("C:\clone\IS808_adv_DS_lab")
data_dir = projectfolder / "data"

# %%
# ! pip install pyarrow

# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules


import io
import json
import pickle
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from tqdm import tqdm
# %%
dir_path = projectfolder / "vis"
""" df_network_voter_edges =pd.read_csv(f"{dir_path}/vis_network_voter_edges.csv")
df_network_voter_edges.info()
df_network_voter_edges.count()
df_network_voter_edges.head()

df_network_voter_edges """

df = pd.read_parquet(data_dir/"regression_frame_merged.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df.head()


# Appendix C: table
df_stats= pd.DataFrame({'Shared voting choices': df['nsharedchoicesnormalized'].describe().apply("{0:.2f}".format),
'Shared DAO participation': df['nshareddaosnormalized'].describe().apply("{0:.2f}".format),
'Category distance': df['categorydistance'].describe().apply("{0:.2f}".format),
'Collection distance': df['collectiondistance'].describe().apply("{0:.2f}".format),
'1st degree similarity': df['firstdegreesimilarity'].describe().apply("{0:.2f}".format),
'2nd degree similarity': df['seconddegreesimilarity'].describe().apply("{0:.2f}".format),
})
df_stats = df_stats.reset_index()
df_stats = df_stats.drop([df_stats.index[0],df_stats.index[2],df_stats.index[3],df_stats.index[7]])
df_stats = df_stats.transpose()
df_stats = df_stats.rename(columns=df_stats.iloc[0]).iloc[1: , :]


with open(f"{data_dir}/var_summary_stats.tex",'w') as tf:
    tf.write(df_stats.to_latex(index=True))