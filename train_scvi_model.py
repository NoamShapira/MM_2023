from typing import Optional

import anndata as ad
import scvi
import torch
from omegaconf import OmegaConf

from io_utils import generate_path_in_output_dir
from sc_classification.var_genes import normalize_and_choose_genes, shuang_genes_to_keep


def train_scvi_model(adata_train: ad.AnnData, counts_layer: str = "counts",
                     batch_key: Optional[str] = None) -> scvi.model.SCVI:
    scvi.model.SCVI.setup_anndata(
        adata_train,
        layer=counts_layer,
        batch_key=batch_key,
    )
    model = scvi.model.SCVI(adata_train, n_latent=20, n_layers=2, dropout_rate=0.2, deeply_inject_covariates=True)

    model.train(batch_size=256, early_stopping=True)
    return model


def generate_model_name(config, extra_description: Optional[str] = None) -> str:
    if config.sc_classification.use_shuang_var_genes != 'None':
        name = f"{config.outputs.scvi_model_prefix}_{config.sc_classification.use_shuang_var_genes}_genes"
    else:
        name = config.outputs.scvi_model_prefix
    if extra_description is not None:
        name += extra_description
    return name


def load_pp_adata_after_norm_and_hvg(config) -> ad.AnnData:
    adata_path = generate_path_in_output_dir(config, config.outputs.processed_adata_file_name, add_version=True)
    adata = ad.read_h5ad(adata_path)
    norm_adata = norm_and_hvg(adata, config)
    return norm_adata


def norm_and_hvg(adata, config):
    counts_layer = config.scvi_settings.counts_layer_name
    adata.layers[counts_layer] = adata.X.copy()  # preserve counts needed for normalize_and_choose_genes
    genes_to_keep = shuang_genes_to_keep(genes_names=adata.var_names,
                                         flavor=config.sc_classification.use_shuang_var_genes)
    norm_adata = normalize_and_choose_genes(adata, config, genes_to_keep=genes_to_keep)
    return norm_adata


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    conf = OmegaConf.load('config.yaml')

    adata_for_training = load_pp_adata_after_norm_and_hvg(conf)

    model = train_scvi_model(adata_for_training,
                             counts_layer=conf.scvi_settings.counts_layer_name,
                             batch_key=conf.scvi_settings.batch_key)

    model_name = generate_model_name(conf)
    model_path = generate_path_in_output_dir(conf, model_name,
                                             add_date_timestamp=conf.outputs.add_timestamp)
    model.save(model_path, overwrite=True)
