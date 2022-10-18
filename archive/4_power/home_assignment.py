#!/usr/bin/env python
# coding: utf-8

# ## DAO Power Analysis
#
#

# In[3]:


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

from etherscan import Etherscan

import json
import time
import os
import random
import math

import scipy.stats as st


print(pd.__version__)
print(np.__version__)
print(requests.__version__)
print(plt.matplotlib.__version__)


# Import authentication key.

# In[4]:


# https://martin-thoma.com/configuration-files-in-python/

import json

with open("../3_api/.private/keys.json") as keys_file:
    KEYS = json.load(keys_file)

# Note: don't print the key, or if you do, delete the cell's output
# (cell outputs are saved and can be sent to Github).


# #### Define functions to call different APIs

# In[5]:


## DEEPDAO


def deepdao(query, params=None, post=False):

    ENDPOINT = "https://api.deepdao.io/v0.1/"

    headers = {"x-api-key": KEYS["DEEPDAO"], "accept": "application/json"}

    if post:
        response = requests.post(ENDPOINT + query, headers=headers, json=params)
    else:
        response = requests.get(ENDPOINT + query, headers=headers, params=params)

    print(response)
    return response.json()


## ETHERSCAN
############


def etherscan(params={}):

    ENDPOINT = "https://api.etherscan.io/api"

    params["apikey"] = KEYS["ETHERSCAN"]

    response = requests.get(
        ENDPOINT,
        headers={"accept": "application/json", "User-Agent": ""},
        params=params,
    )

    print(response)
    return response.json()


eth = Etherscan(KEYS["ETHERSCAN"])

## SNAPSHOT
###########

SNAPSHOT_ENDPOINT = "https://hub.snapshot.org/graphql"

snapshot = Client(transport=AIOHTTPTransport(url=SNAPSHOT_ENDPOINT))


def snapshot_rest(query, params=None):

    response = requests.post(
        SNAPSHOT_ENDPOINT,
        headers={"accept": "application/json"},
        params={"query": query},
    )

    print(response)
    return response.json()["data"]


## THE GRAPH
############

## Endpoints depends on subgraph of interest.


# ## Load DAOs.
#
# Saved in previous lecture.

# In[6]:


## All Daos according to Deepdao
daos_deepdao = pd.read_json("../3_api/deepdao/data/daos_deepdao.json")
daos_deepdao.info()


# In[7]:


# ## Warning: it throws an error.

# ## All Daos according to Snapshot
# daos_snapshot = pd.read_json("../3_api/snapshot/data/daos_snapshot.json")
# daos_snapshot.info()


# Workaround.

# In[8]:


with open("../3_api/snapshot/data/daos_snapshot.json") as file:
    daos_snapshot = json.load(file)

daos_snapshot = pd.DataFrame(daos_snapshot)
daos_snapshot.info()


# ### Updated Utility Functions.
#

# #### Load JSON files into Pandas.
#
# Pandas throws an error if JSON contains too large ints, so we load JSON separately and then feed the Pandas.

# In[9]:


def pd_read_json(file):
    ## Prevents Value too big Error.
    with open(file) as f:
        df = json.load(f)
    df = pd.DataFrame(df)
    return df


# ### Keep GQL queries separated.
#
# GQL queries can be quite cumbersome, so it's better to write them into a separate file and then load them in here.

# In[10]:


def get_query(filename, do_gql=False):
    with open("gql_queries/" + filename.replace(".gql", "") + ".gql") as f:
        query = f.read()
        if do_gql:
            query = gql(query)
    return query


## Alias gq.
gq = get_query


# #### Enhanced API Wrapper
#
# With this wrapper we want to:
# - Save data as we download it in order to avoid data loss in case of an error (remote or local).
# - Save the data periodically and in small chuncks, which is usually easier to load back.
# - Catch errors as much as possible, to terminate gracefully the execution.
# - Execute both GQL and simple rest requests for any type of query.
#

# In[11]:


async def gql_all(
    query,
    field,
    first=1000,
    skip=None,
    initial_list=None,
    counter=True,
    limit=None,
    save=None,
    save_interval=10,
    clear_on_save=False,
    append=True,
    rest=False,
    data_dir="data",
    save_counter=1,
    vars=None,
):

    ## The returned value and the varible used to accumulate results.
    out = []

    ## Utility function to save intermediate and final results.
    def save_json():

        # Pandas has problem load pure json saves.
        # Hence we create a pandas Dataframe and save it.
        # nonlocal append
        # flag = "a" if append else "w"
        # with open("data/" + save, flag) as f:
        #     json.dump(out, f)
        #     print("Saved.")

        nonlocal out
        df = pd.DataFrame(out)

        if clear_on_save:

            nonlocal save_counter

            sv = str(save_counter)
            sv = sv.zfill(5)
            save_counter += 1

            filename = save.replace(".json", "_" + sv + ".json")

            out = []
            out_str = "Saved and cleared."
        else:
            filename = save
            out_str = "Saved."

        df.to_json(data_dir + "/" + filename, orient="records")
        print(out_str)

    ## Load initial list.
    ## If no skip is provided, then skip is set to the length of
    ## the initial list, otherwise we use the user-specified value
    if initial_list:
        out = initial_list
        if skip is None:
            skip = len(out)
    elif skip is None:
        skip = 0

    ## Make a GQL query object, if necessary.
    if not rest and type(query) == str:
        query = gql(query)

    my_counter = 0
    fetch = True
    try:
        while fetch:

            my_counter += 1
            if limit and my_counter > limit:
                print("**Limit reached: ", limit)
                fetch = False
                continue

            if rest:

                # Building query manually.
                q = query.replace("($first: Int!, $skip: Int!)", "")
                q = q.replace("$first", str(first))
                q = q.replace("$skip", str(skip))
                # print(q)

                ## Optional additional variables.
                if vars:
                    for v in vars:
                        q = q.replace("$" + v, str(vars[v]))

                res = snapshot_rest(q)

            else:

                _vars = {"first": first, "skip": skip}

                ## Optional additional variables.
                if vars:
                    _vars = {**_vars, **vars}

                res = await snapshot.execute_async(query, variable_values=_vars)

            if not res[field]:
                print("**I am done fetching!**")
                fetch = False
            else:
                out.extend(res[field])
                skip += first
                if counter:
                    print(my_counter, len(out))

                if save and my_counter % save_interval == 0:
                    save_json()

        if save and my_counter % save_interval != 0:
            save_json()

    except Exception as e:
        print(str(e))
        print("**An error occurred, exiting early.**")
        if save:
            save_json()

    return out


# As we split data from the API in multiple smaller files, we want to have a quick method to load them back into one place.

# In[12]:


def pd_read_dir(dir, blacklist=None, whitelist=None, ext=(".json")):
    dir_df = pd.DataFrame()

    for file in os.listdir(dir):
        if blacklist and file in blacklist:
            continue
        if whitelist and file not in whitelist:
            continue

        if file.endswith(ext):
            tmp_df = pd_read_json(dir + "/" + file)
            dir_df = pd.concat([dir_df, tmp_df])

    return dir_df


# ## Spaces

# We did not fetch all the useful information in the previous lecture. Let's do it again.

# **Exercise: use the functions above to load all spaces**
#
# Important! Try with GQL and with REST.
#
# Remember to save the file.
#
# Try different values of `first`.

# In[13]:


## Your code here.
from re import S


q = get_query(filename="snapshot_spaces")
res = await gql_all(query=q, field="spaces", save="snapshot_spaces.json")
res

# q = get_query(filename="snapshot_followers_of_space")
# res = await gql_all(
#     query=q,
#     field="follows",
#     vars = {
#         "space": "yam.eth"
#     }
# )


# **Exercise: verify that you can import the file containing the spaces that just saved**

# In[14]:


spaces = pd_read_json("data/snapshot_spaces.json")


# **Exercise: plot the relationship between followers and proposals across DAOs**
#
# Question: ...and votes?
# Bonus: jitter the dots, and try log scales.

#

# In[18]:


spaces = pd.DataFrame(res)
spaces.info()
## Your code here.


# ### Checking last week home assignment.
#
# - Pick a small (<1k), medium (<10k), and large  DAO (10k+)
# - Get all members (if possible try different APIs)
# - Order them by wealth in ETH
# - Compute correlation ETH wealth and number of votes in the DAO
#

# Let's see whether our cutpoints make sense.

# In[19]:


def dao_size(x, T1=1000, T2=10000):
    if x < T1:
        return "small"
    if x < T2:
        return "medium"
    return "large"


spaces["size"] = spaces["followersCount"].apply(dao_size)


# In[20]:


spaces["size"].value_counts()


# Most of the DAOs are rather small in size.

# In[21]:


spaces["followersCount"].describe()


# In[22]:


spaces10 = spaces[spaces["followersCount"] >= 10]
spaces10.describe()


# In[23]:


sm = spaces10[spaces10["size"] == "small"].sample(1)
sm[["name", "followersCount", "website"]]


# In[24]:


## id = hashflowdao.eth
DAO_ID = "hashflowdao.eth"
# DAO_ID = sm['id'].iloc[0]


# In[25]:


query_followers = gq("snapshot_followers_of_space")
followers = await gql_all(query_followers, field="follows", vars={"space": DAO_ID})


# In[22]:


len(followers)


# In[26]:


followers = pd.DataFrame(followers)
followers.head()


# In[27]:


print(sum(followers.duplicated()))


# Now we want to get the **wealth** in ETH of the followers.
#
# Etherscan's api is quite handy...
#
# **Exercise: finish the function below**

# In[28]:


from etherscan import Etherscan
import json

with open("../3_api/.private/keys.json") as keys_file:
    KEYS = json.load(keys_file)

APIKEY = KEYS["ETHERSCAN"]
eth = Etherscan(APIKEY)


# In[29]:


def get_eth_wealth(addresses, K=20, limit=None):

    idx = 0
    counter = 0
    eth_wealth = []
    n_addresses = len(addresses)

    while (idx < n_addresses) and (limit is None or counter < limit):

        ## Your code here.
        my_addresses = addresses[idx : idx + K]
        res = eth.get_eth_balance_multiple(my_addresses)
        eth_wealth += res
        idx += K
        counter += 1
        print(counter, idx, len(eth_wealth))

    print("**Got all of them!")
    return eth_wealth


# In[30]:


eth_wealth = get_eth_wealth(followers["follower"])


# In[31]:


len(eth_wealth)


# In[32]:


sm_wealth = pd.DataFrame(eth_wealth)


# In[33]:


sm_wealth.head()


# Get the votes of the space.

# In[34]:


query_votes = gq("snapshot_votes_of_space")
votes = await gql_all(query_votes, field="votes", vars={"space": DAO_ID})


# In[ ]:


# In[36]:


votes = pd.DataFrame(votes)
votes.info()
votes.head()


# In[ ]:


# **Exercise: Unnest proposal id.**

# In[37]:


## Your code here.
votes["space"] = votes.space.map(lambda x: x["id"])
votes["proposal"] = votes.proposal.map(lambda x: x["id"])


# In[38]:


votes.head()


# In[39]:


votes["proposal"].value_counts()


# In[40]:


votes["voter"].value_counts()


# In[41]:


votes["choice"].value_counts()


# We have more votes than followers...\
# After inquirying with Snapshot staff on Discord, **it is possible to vote without following an organization.**\
# Let's get the wealth of all voters.

# In[52]:


eth_wealth2 = get_eth_wealth(votes["voter"])


# In[55]:


eth_wealth2


# In[56]:


sm_wealth = pd.DataFrame(eth_wealth2)
sm_wealth.head()


# In[57]:


votes_wealth = pd.merge(sm_wealth, votes, left_on="account", right_on="voter")
votes_wealth["balance"] = votes_wealth["balance"].astype("float64")
votes_wealth.info()


# In[58]:


df_groups = votes_wealth.groupby("choice")


# In[59]:


df_groups["balance"].describe()


# In[64]:


stats = df_groups["balance"].agg(["mean", "sem"])
stats["ci95_hi"] = stats["mean"] + 1.96 * stats["sem"]
stats["ci95_lo"] = stats["mean"] - 1.96 * stats["sem"]
stats


# In[66]:


fig, axes = plt.subplots(ncols=3, nrows=1)

for c, ax in enumerate(axes.flatten()):
    c += 1
    x = votes_wealth[votes_wealth["choice"] == c]
    ax.boxplot(x["balance"])

    ax.set_title("Choice={}".format(c))

plt.tight_layout()


# #### Messages
#
# We get a better idea about the history of our DAO.

# In[67]:


query = gq("snapshot_messages_of_space")
messages = await gql_all(
    query,
    field="messages",
    # rest=True,
    # save="snapshot_followers_small.json",
    data_dir="data/",
    first=1000,
    vars={"space": DAO_ID},
)


# In[68]:


messages = pd.DataFrame(messages)
messages.info()


# In[69]:


messages.head()


# In[70]:


messages["type"].value_counts()


# The number of votes is larger than what we downloaded and the number of follows smaller. After inquiring with the Snapshot's staff on Discord:
#
# - admins might have deleted/closed/archived some proposals,
# - one should check the data of creation of a dato, as some features of the platform might have been added later.

#

#
