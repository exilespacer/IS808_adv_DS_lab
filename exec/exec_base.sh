#!/bin/sh
#$ -cwd
#$ -pe onenode 2
#$ -l m_mem_free=24G
#$ -m abe
#$ -M yen.chiayi@gmail.com
echo "Starting Job at `date`."

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# miniconda
# export PATH="/home/mannheim/svahlpah/miniconda/bin:$PATH"  # commented out by conda initialize  


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/mannheim/svahlpah/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/mannheim/svahlpah/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/mannheim/svahlpah/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/mannheim/svahlpah/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<