import requests
import pandas as pd
import numpy as np



def snapshot_api(query, params=None):
    ENDPOINT = 'https://hub.snapshot.org/graphql'
    response = requests.post(ENDPOINT,
                            headers={                      
                                'accept': 'application/json'
                            },
                            params={
                                'query': query
                            })

    print(response)
    return response.json()['data']

