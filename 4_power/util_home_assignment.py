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

    ENDPOINT = 'https://api.deepdao.io/v0.1/'

    headers={
        'x-api-key': KEYS['DEEPDAO'],
        'accept': 'application/json'
    }

    if post:
        response = requests.post(ENDPOINT + query,
                                headers=headers,
                                json=params)
    else:
        response = requests.get(ENDPOINT + query,
                                headers=headers,
                                params=params)

    print(response)
    return response.json()

## ETHERSCAN
############

def etherscan(params={}):

    ENDPOINT = 'https://api.etherscan.io/api'

    params['apikey'] = KEYS['ETHERSCAN']

    response = requests.get(ENDPOINT,
                            headers={
                                'accept': 'application/json',
                                "User-Agent": ""
                            },
                            params=params)

    print(response)
    return response.json()


eth = Etherscan(KEYS['ETHERSCAN'])

## SNAPSHOT
###########

SNAPSHOT_ENDPOINT = "https://hub.snapshot.org/graphql"

snapshot = Client(
    transport=AIOHTTPTransport(url=SNAPSHOT_ENDPOINT)
)


def snapshot_rest(query, params=None):

    response = requests.post(SNAPSHOT_ENDPOINT,
                            headers={                      
                                'accept': 'application/json'
                            },
                            params={
                                'query': query
                            })

    print(response)
    return response.json()['data']

## THE GRAPH
############

## Endpoints depends on subgraph of interest.


# ## Load DAOs.
# 

def pd_read_json(file):
    ## Prevents Value too big Error.
    with open(file) as f:
        df = json.load(f)
    df = pd.DataFrame(df)
    return df


def get_query(filename, do_gql=False):
    with open("gql_queries/" + filename.replace(".gql", "") + ".gql") as f:
        query = f.read()
        if do_gql: query = gql(query)
    return query
    
## Alias gq.
gq = get_query

async def gql_all(query, field, first=1000, skip=None, initial_list=None, 
                  counter = True, limit=None, save=None, save_interval=10, clear_on_save = False, append=True, rest=False, data_dir="data", save_counter = 1, vars=None):

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

            filename = save.replace('.json', '_' + sv + '.json')
            
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
                print('**Limit reached: ', limit)
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
                    _vars ={**_vars, ** vars}

                res = await snapshot.execute_async(query, variable_values=_vars)
            
            if not res[field]:
                print('**I am done fetching!**')
                fetch = False
            else:
                out.extend(res[field])
                skip += first
                if counter: print(my_counter, len(out))

                if save and my_counter % save_interval == 0:
                    save_json()

        if save and my_counter % save_interval != 0:
            save_json()

    except Exception as e:
        print(str(e))
        print("**An error occurred, exiting early.**")
        if save: save_json()
    
    return out



def pd_read_dir(dir, blacklist=None, whitelist=None, ext=('.json')):
    dir_df = pd.DataFrame()
    

    for file in os.listdir(dir):
        if blacklist and file in blacklist:
            continue
        if whitelist and file not in whitelist:
            continue

        if file.endswith(ext):
            tmp_df = pd_read_json(dir + '/' + file)
            dir_df = pd.concat([dir_df, tmp_df])
    
    return dir_df


from etherscan import Etherscan
import json

with open("../3_api/.private/keys.json") as keys_file:
    KEYS = json.load(keys_file)

APIKEY = KEYS['ETHERSCAN']
eth = Etherscan(APIKEY)


# In[29]:



def get_eth_wealth(addresses, K=20, limit=None):
    
    idx = 0
    counter = 0
    eth_wealth = []
    n_addresses = len(addresses)
    
    while (idx < n_addresses) and (limit is None or counter < limit):
        
        ## Your code here.
        my_addresses = addresses[idx: idx+K]
        res = eth.get_eth_balance_multiple(my_addresses)
        eth_wealth += res
        idx += K
        counter += 1
        print(counter, idx, len(eth_wealth))
    
    print('**Got all of them!')
    return eth_wealth


