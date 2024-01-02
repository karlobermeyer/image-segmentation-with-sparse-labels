#!/bin/bash
# This script sets up your environment for running notebooks and Python scripts.
# Run `$ source ./setenv.sh` from the repo root directory every time you open a
# new shell.


# Make sure you don't accidentally commit credentials.
git update-index --assume-unchanged ./.env


echo -n 'Loading credentials from `.env`...'
#source .env  # May not work on `.env` files with comments.
export $(grep -v '^#' ./.env | xargs)  # Works with comments.
echo "done."


# Let all packages in subdirectories see each other.
echo -n "Setting environment variables..."
project_directory=`pwd`
export PYTHONPATH=$PYTHONPATH:${project_directory}/src/
export REPO_ROOT=${project_directory}
export CITYSCAPES_DATASET=${project_directory}/data/cityscapes/
echo "done."


# Activate Python virtual environment, creating it from `requirements.txt` if it
# doesn't already exist.
venv_dirname="image-segmentation-with-sparse-labels-env"
if [ -d "${venv_dirname}" ]; then
    echo -n "Activating Python virtual environment..."
    source ${venv_dirname}/bin/activate
    echo "done."
else
    echo -n "Creating Python virtual environment..."
    python -m venv ${venv_dirname}
    echo "done."
    echo -n "Activating Python virtual environment..."
    source ${venv_dirname}/bin/activate
    echo "done."
    echo "Installing dependencies (may take a few minutes)..."
    for i in {5..1}; do
        echo -ne " Beginning in $i seconds. \r"
        sleep 1
    done
    echo ""
    pip install -r requirements.txt
    #pip install -U -r requirements.txt  # Install+upgrade.
    echo "Finished installing dependencies."
fi


echo -n "Redefining grep* search commands..."
# Redefine search commands so that they exclude directories `EXCLUDE_DIRNAMES`
# when run from the repo root directory.
EXCLUDE_DIRNAMES=(
    "${project_directory}/data"
    "${project_directory}/models"
    "${project_directory}/logs"
    ".git"
    ".ipynb_checkpoints"
    "notebooks/.ipynb_checkpoints"
    "__pycache__"
    "image-segmentation-with-sparse-labels-env"
)
# ::WARNING:: When using any grep aliases that have an ``--include'' flag, must
# use `./` instead of `*` at the end, e.g.,
# `$ grep --include=*.py "blah" ./`
function grepn() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHnI --color $exclude_option "$1" ./
}
function grepni() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHinI --color $exclude_option "$1" ./
}
function grepy() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn --color $exclude_option --include=*.py --include=*.pyx "$1" ./
}
function grepyj() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn --color $exclude_option --include=*.ipynb "$1" ./
}
function grepya() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn --color $exclude_option --include=*.py --include=*.pyx --include=*.ipynb "$1" ./
}
function greprs() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn --color $exclude_option --include=*.rs "$1" ./
}
function grepc() {
        exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn $exclude_option --include=*.h --include=*.hpp --include=*.c\
     --include=*.cpp --include=*.cc --include=*.C "$1" ./
}
function grepmd() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn --color $exclude_option --include=*.md "$1" ./
}
function grept() {
    exclude_option=$(printf -- "--exclude-dir=%s " "${EXCLUDE_DIRNAMES[@]}")
    grep -sRHn --color $exclude_option --include=*.text --include=*.txt "$1" ./
}
echo "done."
