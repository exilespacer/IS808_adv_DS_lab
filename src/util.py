import requests
import pandas as pd
import numpy as np
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import json


def snapshot_api(query, params=None):
    ENDPOINT = "https://hub.snapshot.org/graphql"
    response = requests.post(
        ENDPOINT, headers={"accept": "application/json"}, params={"query": query}
    )

    print(response)
    return response.json()["data"]


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

    return response.json()["data"]


def gq(filename):
    with open(filename) as f:
        query = f.read()
    return query


async def gql_all(
    query: str,
    field: str,
    batch_size: int = 1000,
    skip: int = None,
    initial_list=None,
    counter: bool = True,
    limit: int = None,
    output_filename=None,
    save_interval: int = 10,
    clear_on_save: bool = False,
    rest: bool = False,
    output_dir="data",
    save_counter: int = 1,
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

        if clear_on_save is True:

            nonlocal save_counter

            sv = str(save_counter)
            sv = sv.zfill(5)
            save_counter += 1

            filename = output_filename.replace(".json", "_" + sv + ".json")

            out = []
            out_str = "Saved and cleared."
        else:
            filename = output_filename
            out_str = "Saved."

        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_json(output_dir / filename, orient="records")
        print(out_str)

    ## Load initial list.
    ## If no skip is provided, then skip is set to the length of
    ## the initial list, otherwise we use the user-specified value
    if initial_list is not None:
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
            if limit is not None and my_counter > limit:
                print("**Limit reached: ", limit)
                fetch = False
                continue

            if rest is True:

                # Building query manually.
                q = query.replace("($first: Int!, $skip: Int!)", "")
                q = q.replace("$first", str(batch_size))
                q = q.replace("$skip", str(skip))
                # print(q)

                ## Optional additional variables.
                if vars is not None:
                    for v in vars:
                        q = q.replace("$" + v, str(vars[v]))

                res = snapshot_rest(q)

            else:

                _vars = {"first": batch_size, "skip": skip}

                ## Optional additional variables.
                if vars is not None:
                    _vars = _vars | vars

                res = await snapshot.execute_async(query, variable_values=_vars)

            if len(res[field]) == 0:
                print("**I am done fetching!**")
                fetch = False
            else:
                out.extend(res[field])
                skip += batch_size
                if counter is True:
                    print(my_counter, len(out))

                if output_filename is not None and my_counter % save_interval == 0:
                    save_json()

        if output_filename is not None and my_counter % save_interval != 0:
            save_json()

    except Exception as e:
        print(str(e))
        print("**An error occurred, exiting early.**")
        if output_filename is not None:
            save_json()
        raise

    return out


def pd_read_json(file):
    ## Prevents Value too big Error.
    with open(file) as f:
        df = json.load(f)
    df = pd.DataFrame(df)
    return df
