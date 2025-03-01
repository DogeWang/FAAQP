# FAAQP

FAAQP is a data-driven AQP method that achieves state-of-the-art performance in approximate query processing (AQP). The structure of BSPN is an extension of existing SPN-based methods. This is a version of BSPN based on RSPN (https://github.com/DataManagementLab/deepdb-public/tree/master).

Hanbing Zhang, Yinan Jing, Zhenying He, Kai Zhang, and X. Sean Wang: "FAAQP: Fast and Accurate Approximate Query Processing based on Bitmap-augmented Sum-Product Network", SIGMOD 2025.

# Setup
Tested with python3.7 and python3.8
```
git clone git@github.com:DogeWang/SPNPP.git
cd SPNPP
sudo apt install -y libpq-dev gcc python3-dev
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

For python3.8: Sometimes spflow fails, in this case remove spflow from requirements.txt, install them and run
```
pip3 install spflow --no-deps
```
# AQP
## Flights pipeline
Download data (5 million tuples) from https://www.transtats.bts.gov/Homepage.asp

Generate hdf files from csvs.
```
python3 maqp.py --generate_hdf
    --dataset flights_origin
    --csv_seperator ,
    --csv_path ../flights-benchmark
    --hdf_path ../flights-benchmark/gen_hdf
```

Learn the ensemble.
```
python3 maqp.py --generate_ensemble 
    --dataset flights_origin
    --samples_per_spn 10000000 
    --ensemble_strategy single 
    --hdf_path ../flights-benchmark/gen_hdf 
    --ensemble_path ../flights-benchmark/spn_ensembles
    --rdc_threshold 0.3
    --post_sampling_factor 10
```

Compute ground truth by using PostgreSQL
```
python3 maqp.py --aqp_ground_truth
    --dataset flights_origin
    --query_file_location ./benchmarks/flights/sql/aqp_test_queries.sql
    --target_path ./benchmarks/flights/aqp_test_queries_ground_truth.pkl
    --database_name flights_origin   
```

Evaluate the AQP queries.
```  
python3 maqp.py --evaluate_aqp_queries
    --dataset flights_origin
    --target_path ./baselines/aqp/results/deepDB/aqp_test_queries_BSPN.csv
    --ensemble_location ../flights-benchmark/spn_ensembles/ensemble_single_flights_origin_5000000_RoaringBitmap.pkl
    --query_file_location ./benchmarks/flights/sql/aqp_test_queries.sql
    --ground_truth_file_location ./benchmarks/flights/aqp_test_queries_ground_truth.pkl
```
