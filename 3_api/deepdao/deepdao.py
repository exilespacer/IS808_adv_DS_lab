# %%
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd.__version__)
print(np.__version__)
print(requests.__version__)
print(plt.matplotlib.__version__)

# %%
# https://martin-thoma.com/configuration-files-in-python/

import json

with open("../../.private/keys.json") as keys_file:
    KEYS = json.load(keys_file)

APIKEY = KEYS["DEEPDAO"]
# Note: don't print the key, or if you do, delete the cell's output
# (cell outputs are saved and can be sent to Github).
ENDPOINT = "https://api.deepdao.io/v0.1/"

# %%
def api(query, params=None, post=False):

    # https://datagy.io/python-requests-authentication/
    headers = {"x-api-key": APIKEY, "accept": "application/json"}

    if post:
        response = requests.post(ENDPOINT + query, headers=headers, json=params)
    else:
        response = requests.get(ENDPOINT + query, headers=headers, params=params)

    print(response)
    return response.json()


# %%
organizationId = "529571a8-5816-47ff-a50e-1e6372e2324b"
query = f"organizations/{organizationId}/top_active_in_organization"
params = {"orderBy": "votesCount"}
res = api(query, params=params)

# %%
daomemberaddresses = pd.DataFrame(res["data"])["address"].to_list()
# %%
