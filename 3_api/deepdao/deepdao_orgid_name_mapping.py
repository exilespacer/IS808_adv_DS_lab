# %%
from pathlib import Path
import json

# %%
projectfolder = Path("/project/IS808_adv_DS_lab")
folder = projectfolder / "3_api/deepdao/data"

# %%

f = folder / "daos_deepdao.json"
with open(f, "r") as fp:
    data = json.load(fp)

# %%
mapping = {dao["organizationId"]: dao["name"] for dao in data}

# %%
of = folder / "deepdao_id_name_mapping.json"
with open(of, "w") as fp:
    json.dump(mapping, fp)
