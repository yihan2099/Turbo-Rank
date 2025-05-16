# 1. Make sure pip itself is up to date
python -m pip install --upgrade pip

# 2. Upgrade all installed packages in one go
pip install --upgrade $(pip freeze | awk -F '==' '{print $1}')

# 3. Save the updated requirements.txt
pip freeze > requirements.txt