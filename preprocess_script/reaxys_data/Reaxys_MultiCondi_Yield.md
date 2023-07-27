# Reaxys-MultiCondi-Yield Curation

The original data of this dataset comes from about 17,000 total synthetic literatures. Use the DOI number to extract the corresponding reaction data from Reaxys in batches. See `Egret/preprocess_script/reaxys_data/qurey_file` for batch query scripts. Combine all extracted data and  save to `Egret/dataset/source_dataset/Reaxys-MultiCondi-Yield/reaxys_total_syn_yield_data.csv` The csv header is as follows:
```
'Reaction ID',
'Record Type',
'Reactant',
'Product',
'Reaction',
'Reaction Details: Reaction Classification',
'Time (Reaction Details) [h]',
'Temperature (Reaction Details) [C]',
'Pressure (Reaction Details) [Torr]',
'pH-Value (Reaction Details)',
'Other Conditions',
'Reaction Type',
'Product.1',
'Yield',
'Yield (numerical)',
'Yield (optical)',
'Reagent',
'Catalyst',
'Solvent (Reaction Details)',
'References',
```

Then:
```
cd Egret/preprocess_script/reaxys_data
python 1.0_preliminary data cleaning.py
python 1.1.convert_condition_name2smiles.py
python 1.2.get_reaxys_multicondi_yield.py
python 1.3.split_dataset.py
```
We provide the "Reaction ID" and "Links to Reaxys" for all the reactions in the training, validation, and test sets of Reaxys_Multicondi_Yield(`Egret/dataset/source_dataset/Reaxys-MultiCondi-Yield`). Readers can effortlessly reproduce our dataset by using the "Reaction ID" and "Links to Reaxys" to export these reactions from the Reaxys Database.