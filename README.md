# SPN++

SPN++ is a data-driven learned database component achieving state-of-the-art-performance in cardinality estimation and 
approximate query processing (AQP). The structure of SPN++ is an extension of existing SPN-based methods. This is a version of SPN++ based on RSPN (https://github.com/DataManagementLab/deepdb-public/tree/master).


# Setup
Tested with python3.7 and python3.8
```
git clone https://github.com/DataManagementLab/deepdb-public.git
cd deepdb-public
sudo apt install -y libpq-dev gcc python3-dev
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

For python3.8: Sometimes spflow fails, in this case remove spflow from requirements.txt, install them and run
```
pip3 install spflow --no-deps
```
