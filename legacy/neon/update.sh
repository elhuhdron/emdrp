
# need to activate anaconda python2 first!!!
# ac2init
export PATH="/home/watkinspv/anaconda2/bin:$PATH"
source deactivate
conda remove --name neon --all
conda info --envs
cd ~/gits/neon
git pull
conda create --name neon pip
source activate neon
make sysinstall -e VIS=true
cd ../emdrp/neon/
pip install -r requirements.txt

