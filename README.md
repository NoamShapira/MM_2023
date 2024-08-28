# MM_2023

configuration in config.yaml
you can edit the file to run with different config, most scripts get the conf file as an argument
so it is possible to save current configurations

scripts will create intermediate .h5ad files to /outputs dir (as shown in config)
the names of outputs are defined in the config
you may need to create one (the output dir) at first run

raw data pipeline until full annotation
0.  create updated plates tables and update config:
   1. split_plates_metadata_excel_to_csvs - this is a convenient script to create the csvs used in the config
   2. update config:
      - after running this script you need to manually edit the config to the csv created
      - also update data_loading.version in order for the scripts to create new output files instead of overriding the old
      - check other relevant parameters updates (all scripts should work without further updates)
1. load_sc_data_to_anndata
2. pp_adata
3. infer annotation (also do clustering)

model training scripts:
1. train_scvi_model - used to create the model used to classify TME vs PC, and TME sub clusters
   - if is run, need to update the config in order for the infer annotation script to use the new model
2. train_scvi_model_on_sub_population - used to create the model used to embed and cluster PC 
   - if is run, need to update the config in order for the infer annotation script to use the new model
   - this script is versatile and can be used for other purposes

after running this script follow notebook re_annotate_mm-scVi_PC_sub_pop.ipynb to create final PC annotations, 
also in this notebook We update the Disease column using the clinical data from hospital

most analysis is done in notebooks using .h5ad files created by the pipeline
