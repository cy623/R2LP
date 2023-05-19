# RoGNN
This is the official repository of paper "Robust Graph Learning against Arbitrary Heterophily and Label Noise".

Setup
-------
* torch==1.12.1
* networkx==2.6.3
* scipy==1.7.3
* numpy==1.21.5

Repository structure
--------
```python
├── data  # old datasets, including cora, citeseer, pubmed
├── new_data # new datasets, including texas, wisconsin, cornell, actor and chameleon
├── splits # Split part of dataset
├── train.py # the main code
├── model.py # model implementations
├── utils.py # preprocessing subroutines
├── run.sh # run results
```

Run pipeline
--------
1. You can run RoGNN by the script:
```
python RoGNN/train.py --dataset cora
```
2. For more experiments running details, you can ref the running .sh by the script:
```
sh RoGNN/run.sh
```
