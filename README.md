# IS808_adv_DS_lab

## Things we should agree upon
- Use Jupyter notebooks or plain Python files?

## Pre-commit hooks

I added pre-commit hooks that
- format the code -> Standardized code that is readable
- clear the outputs of the jupyter notebooks -> no data leakage
    - Maybe we need to remove this later when we're sharing results, but for now we can keep it in

The installation is simple: In a terminal in the git root folder, execute: `pip install pre-commit && pre-commit install`
