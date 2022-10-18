#!/usr/bin/env python
# coding: utf-8

# ## Theoretical Power

# In[1]:


import itertools
import numpy as np


# Define weights and quorum

# In[2]:


## Number of votes necessary to approve initiative.
QUORUM = 10

## Weights of each player.
weights = [1, 5, 7, 3]
N = len(weights)

## Index of players (from 0 to N-1)
players = range(N)


# #### Shapley Index
#
# "The power of a coalition (or a player) is measured by the fraction of the possible voting sequences in which that coalition casts the deciding vote, that is, the vote that first guarantees passage or failure."
#
# https://en.wikipedia.org/wiki/Shapley%E2%80%93Shubik_power_index
#
# _Note:_ The key word is "sequences".

# **Exercise: Compute the Shapley index for all players.**
#
# _Note:_ `itertools` is your friend.

# In[3]:


## Your code here.


# #### Banzhaf Index
#
# "To calculate the power of a voter using the Banzhaf index, list all the winning coalitions, then count the critical voters. A critical voter is a voter who, if he changed his vote from yes to no, would cause the measure to fail."
#
# https://en.wikipedia.org/wiki/Banzhaf_power_index

# **Exercise: Compute the Banzhaf index for all players.**
#

# In[4]:


## Your code here.


# ## Empirical Power
#
# Potential and exercised.
# Based on "Voting Behavior and Power in Online Democracy: A Study of LiquidFeedback in Germany's Pirate Party" by Kling et al. 2015.

# In[5]:


QUORUM = 10

## Here we have both weights and votes.
weights = [1, 5, 7, 3]
votes = [1, 1, 1, -1]

N = len(weights)
players = range(N)

n_infavor = votes.count(1)
n_against = votes.count(-1)
n_abstained = N - n_infavor - n_against

print(n_infavor, n_against, n_abstained)

## Summing total weights in favor (positive)
## and against (negative).
Wp = 0
Wn = 0

for i in range(N):
    if votes[i] == 1:
        Wp += weights[i]
    elif votes[i] == -1:
        Wn += weights[i]

print(Wp, Wn)


# #### Potential Power
#
# Potential power tests if a weight _i_ is larger than the distance to quorum of all remaining votes:
#
# It is equal to:
#
# - 1, if test is positive;
# - 0, if test is negative

# **Exercise: Compute potential power for a given player.**
#
# _Note_: in the paper they use quorum as a fraction, but you can use it as a count.

# In[6]:


## Your code here.


# #### Exercised Power
#
# Exercised power tests if the outcome of a voting is different with and without a player _i_:
#
# It is equal to:
#
# - 1, if test is positive (outcomes differ);
# - 0, if test is negative (same outcome)

# **Exercise: Compute exercised power for a given player.**
#
# _Note_: in the paper they use quorum as a fraction, but you can use it as a count.

# In[7]:


## Your code here.


# In[ ]:
