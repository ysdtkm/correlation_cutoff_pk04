set -e

mkdir -p data image offline

python2 Py/main.py
python2 Py/plot.py
# python3 Py/offline.py
# python3 Py/offline_plot.py
