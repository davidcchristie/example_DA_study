wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b  -p ./miniconda -f
source miniconda/bin/activate
python -m pip install -r requirements.txt
git clone -b release/v0.1.0 git@github.com:xsuite/tree_maker.git
python -m pip install -e tree_maker
