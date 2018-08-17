set -e

mkdir -p data image offline

python3 Py/main.py
python3 Py/plot.py
python3 Py/offline.py
python3 Py/offline_plot.py
