import config as cfg 
dir_local = cfg.projectfolder / 'wrds'

import os
env_name = os.environ.get('ENV_NAME', 'IS808')
email = os.environ.get('EMAIL', 'yen.chiayi@gmail.com')

import io
from src.i042regression import load_specifications, add_filenames

def get_filename(spec_id): 
    filename = f'exec_regression_spec_{spec_id}.sh'
    path = dir_local / filename
    return path

def create_bash(spec_id):
    filename_input = dir_local / 'exec_base.sh'
    filename_output = get_filename(spec_id)
    try:
        os.system(f'rm {filename_output}*')
    except:
        pass

    # get basic regression file
    with io.open(filename_input, mode = 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    
    # replace email
    idx_email = list(filter(lambda x: lines[x].startswith('#$ -M '), range(len(lines))))[0]
    lines[idx_email] = f'#$ -M {email}\n'

    # add command
    new_lines = lines + [
        '\n',
        f'conda activate {env_name};\n',
        f"time(SPEC_ID={spec_id} python -m src.i042regression) &> log/i042regression_{spec_id}.log;\n"
    ]
    print(new_lines)
    print(filename_output)
    with io.open(filename_output, mode='w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f'created {filename_output}')

def main():
    add_filenames()
    specs = load_specifications()

    # create bash scripts for every spec
    for spec_id in range(len(specs)):
        create_bash(spec_id)
        
    # submit tasks to WRDS server
    for spec_id in range(len(specs)):
        print(specs[spec_id])
        exec_name =  str(get_filename(spec_id)).split(str(cfg.projectfolder))[-1][1:]
        os.system(f'qsub {exec_name}')
        print(f'submitted {exec_name}')

if __name__ == '__main__':
    # ENV_NAME=IS808 EMAIL=yen.chiayi@gmail.com python -m util.create_exec_regression
    main()