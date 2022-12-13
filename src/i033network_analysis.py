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
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules


import io
import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

import matplotlib.pyplot as plt


# %% [markdown]
# # Data for network visualization
# - Source
# - Target
# - weight


# %%
dir_path = projectfolder / "vis"
# !pip install tqdm
# !pip install pyvis


# %% [markdown]
# # Voter network
df_network_voter_edges =pd.read_csv(f"{dir_path}/vis_network_voter_edges.csv")
df_network_voter_edges.info()
df_network_voter_edges.count()
df_network_voter_edges.head()
print (df_network_voter_edges.dtypes)

df_network_voter_nodes =pd.read_csv(f"{dir_path}/vis_network_voter_nodes.csv")
df_network_voter_nodes.count()
df_network_voter_nodes.drop_duplicates('taste')
df_network_voter_nodes.head()
print (df_network_voter_nodes.dtypes)


# %% full
G_voter = nx.from_pandas_edgelist(
    df_network_voter_edges, source="voter1", target="voter2", edge_attr= True,
)
#  
list(G_voter.nodes)
list(G_voter.edges)
G_voter.nodes

for i in sorted(G_voter.nodes()):
    G_voter.nodes[i]['taste'] = df_network_voter_nodes.taste[i]

# nx.set_node_attributes(G_voter, pd.Series(df_network_voter_nodes.taste, index=df_network_voter_nodes.voterid).to_dict(), 'taste')
node_colors = {
    'art': 'blue', 
    'domain-names': 'red', 
    'trading-cards': 'yellow', 
    'virtual-worlds': 'pink',
    'collectibles': 'green',
    'photography-category': 'black',
    'sports': 'purple'
    }

df_network_voter_nodes['node_color'] = df_network_voter_nodes['taste'].map(node_colors)
node_attr = df_network_voter_nodes.set_index('voterid').to_dict(orient = 'index')
nx.set_node_attributes(G_voter, node_attr)

plt.figure(figsize=(6,6))
nx.draw(G_voter,
    pos = nx.spring_layout(G_voter, weight = 'nshareddaos_y'),
    node_size = 200,
    node_color =[G_voter.nodes[n]['node_color'] for n in G_voter.nodes],
    width = [G_voter.edges[e]['nshareddaos_y'] for e in G_voter.edges],
    with_labels = False
    )
plt.plot()
plt.show()

nx.write_gexf(G_voter,f"{dir_path}/gragh_voter.gexf")
with io.open(f"{dir_path}/gragh_voter.nx", mode="wb") as f:
    pickle.dump(G_voter, f)

# %%
net_voter = Network(height="1200px", width="100%", notebook=True)
net_voter.repulsion()
net_voter.from_nx(G_voter)
net_voter.show(f"{dir_path}/pyvis_nx_voter.html")