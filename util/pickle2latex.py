
from src.i042regression import regression_results_folder, pickle2latex
from tqdm import tqdm

for reg in tqdm(regression_results_folder.glob("*.pickle")):
    pickle2latex(reg)