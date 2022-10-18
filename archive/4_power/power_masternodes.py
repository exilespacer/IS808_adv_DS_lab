#!/usr/bin/env python
# coding: utf-8

# ## Power Masternodes
#
# First, load everthing at once.

# In[1]:


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


# https://martin-thoma.com/configuration-files-in-python/

import json

with open("../3_api/.private/keys.json") as keys_file:
    KEYS = json.load(keys_file)

# Note: don't print the key, or if you do, delete the cell's output
# (cell outputs are saved and can be sent to Github).


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


def pd_read_json(file):
    ## Prevents Value too big Error.
    with open(file) as f:
        df = json.load(f)
    df = pd.DataFrame(df)
    return df


def get_query(filename, do_gql=False):
    with open("gql_queries/" + filename.replace(".gql", "") + ".gql") as f:
        query = f.read()
        if do_gql:
            query = gql(query)
    return query


## Alias gq.
gq = get_query


def get_query(filename, do_gql=False):
    with open("gql_queries/" + filename.replace(".gql", "") + ".gql") as f:
        query = f.read()
        if do_gql:
            query = gql(query)
    return query


## Alias gq.
gq = get_query


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
                    _vars = _vars | vars

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


# ## Preparing to compute power as in Mosley et al. (2022).
#
# "Towards a systemic understanding of blockchain governance in proposal voting: A dash case study."

# Load `spaces`, `proposals`, and `votes`.

# In[11]:


spaces = pd_read_json("data/snapshot_spaces.json")

all_proposals = pd_read_json("data/5_snapshot_pancake_proposals.json")


# In[14]:


## If downloaded already.
all_votes = pd_read_dir("data/votes")


# In[15]:


## Otherwise.
## This query takes a while...
# votes_query = gq("snapshot_votes")
# res = await gql_all(votes_query,
#                     field="votes",
#                     rest=True,
#                     save="snapshot_votes_test.json",
#                     data_dir="data/votes/",
#                     save_interval = 20,
#                     limit=2,
#                     first=20000, # First can be a high number.
#                     clear_on_save=True
#                     )


# In[16]:


print("spaces: ", len(spaces))
print("proposals: ", len(all_proposals))
print("votes: ", len(all_votes))


# ### Cleanup

# #### Proposals

# In[17]:


all_proposals.info()


# In[28]:


# all_proposals['space'] = all_proposals['space'].apply(lambda x : x['id'])


# In[29]:


all_proposals["space"].head()


# #### Votes.

# In[30]:


all_votes.head()


# In[31]:


# all_votes['space'] = all_votes['space'].apply(lambda x : x['id'])


# In[32]:


# ## Returns an error, we need to account for a None field.
# # all_votes['proposal'] = all_votes['proposal'].apply(lambda x : x['id'])

# all_votes['proposal'] = all_votes['proposal'].apply(lambda x :
#     x if x is None else x['id']
# )


# #### Pancake Swap

# Who did most of the proposals?

# In[33]:


most_props = spaces[spaces["proposalsCount"] == max(spaces["proposalsCount"])]
DAO_MOST_PROPS_ID = most_props["id"].iloc[0]
DAO_MOST_PROPS_ID


# In[34]:


all_proposals["space"].value_counts()


# In[35]:


pancake_props = all_proposals[all_proposals["space"] == DAO_MOST_PROPS_ID]
pancake_props.info()


# In[36]:


pancake_votes = all_votes[all_votes["space"] == DAO_MOST_PROPS_ID]
pancake_votes.info()


# In[37]:


## Generate an error, there are mixed types.
pancake_votes["choice"].value_counts()


# In[38]:


pancake_votes["choice"].describe()


# Let's remove non 'int' votes (e.g., ranked choices).
#
# In the real analysis we should try to analyze all data.

# In[39]:


print(len(pancake_votes))
pancake_votes = pancake_votes[pancake_votes["choice"].isin([1, 2, 3])]
print(len(pancake_votes))


# In[40]:


pancake_votes["choice"].value_counts()


# Let's center them around zero.

# In[41]:


pancake_votes["choice"] = pancake_votes["choice"] - 2
pancake_votes["choice"].value_counts()


# How much a node deviate from the others in every proposal.

# In[42]:


def euclid(row):
    vote = row["choice"]
    proposal = row["proposal"]
    other_votes = pancake_votes[pancake_votes["proposal"] == proposal]
    distances = other_votes["choice"].apply(lambda x: math.pow((vote - x), 2))
    return math.sqrt(sum(distances))


pancake_votes["vote_distance"] = pancake_votes.apply(euclid, axis=1)


# In[43]:


pancake_votes["vote_distance"].describe()


# In[44]:


pancake_votes["vote_distance"].plot.hist(bins=100)


# In[45]:


pancake_prop_groups = pancake_votes.groupby("proposal")


# What are the proposals with the highest variation in voting?

# In[46]:


pancake_prop_groups["vote_distance"].describe().sort_values("mean", ascending=False)


# **Exercise: implement the Masternode Voting Network algorithm**

# In[47]:


voters = pancake_votes["voter"].unique()
len(voters)


# In[48]:


## Your code here.


# In[ ]:
