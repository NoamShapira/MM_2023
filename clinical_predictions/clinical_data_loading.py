import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_loading.utils import load_dataframe_from_file


def merge_transcriptom_data_to_raw_hospital(transcriptome_dataset: pd.DataFrame,
                                            raw_hospital_dataset: pd.DataFrame,
                                            filter_transcriptome_dataset_by_col: Optional[str] = "Transcriptom",
                                            transcriptome_dataset_patient_id_col_name: Optional[
                                                str] = "PID") -> pd.DataFrame:
    dataset = transcriptome_dataset
    if filter_transcriptome_dataset_by_col is not None:
        dataset = transcriptome_dataset[transcriptome_dataset[filter_transcriptome_dataset_by_col].fillna(False)]
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop(columns=["Unnamed: 0"])
    if transcriptome_dataset_patient_id_col_name is not None:
        dataset = dataset.set_index("PID")

    # add post treatment columns
    post_treatment_cols = [col for col in raw_hospital_dataset.columns if ".2" in col]
    # add fish_columns
    fish_cols = [col for col in raw_hospital_dataset.columns if
                 "t(" in col or "del(" in col or col in ['1q21+', 'IGH rearrangement',
                                                         'Cytogenetics Risk (0=standard risk, 1=single hit, 2=2+ hits)']]
    post_treatment_data = raw_hospital_dataset[raw_hospital_dataset["Time"] != "Post"][
        # TODO not good to throw "Post", nned maybe
        ["Code"] + post_treatment_cols + fish_cols].set_index("Code")
    dataset = dataset.merge(post_treatment_data, how="left", left_index=True, right_index=True, validate="one_to_one")

    # add data to TAL_3 patients
    TAL3_patients = [code for code in dataset.index if "P" == code[0]]
    dataset.loc[TAL3_patients, "Stage"] = 3
    dataset.loc[TAL3_patients, "Lenalidomide"] = 2

    return dataset


def generate_refracrotines_dataset(dataset: pd.DataFrame, treatment: str, non_ref_policy: str, feats: List[str]):
    pre_biopsy_ref_mask = dataset[treatment] == 2
    post_biopsy_ref_mask = dataset[f"{treatment}.2" if treatment != "DARA" else "Daratumumab.2"] == 2
    ref_mask = pre_biopsy_ref_mask | post_biopsy_ref_mask

    NDMM_STAGE_LIST = [1, 2]
    if non_ref_policy == "NDMM":
        newly_diagnosed = dataset["Stage"].apply(lambda x: x in NDMM_STAGE_LIST)
        non_ref_mask = newly_diagnosed
    elif non_ref_policy == "NDMM-POST_TREATMENT_REF":
        newly_diagnosed = dataset["Stage"].apply(lambda x: x in NDMM_STAGE_LIST)
        non_ref_mask = (newly_diagnosed) | (dataset[f"{treatment}.2"] == 4)
    elif non_ref_policy == "NON_EXPOSED":
        non_ref_mask = dataset[treatment].isna()
    else:
        raise NotImplementedError

    X = pd.concat([dataset[ref_mask][feats], dataset[non_ref_mask][feats]], axis=0)
    y = pd.concat([ref_mask[ref_mask].astype(int), non_ref_mask[non_ref_mask].astype(int) - 1])

    return X, y


DEFAULT_RESPONSE_CODING = {
    "response": "R",
    "no_response": "NR",
    "no_data": None
}


def add_response_columns_to_drug_combination(dataset: pd.DataFrame, combination: str,
                                             response_policy: str, no_response_policy: str,
                                             coding: Optional[dict] = None,
                                             raise_errors=True) -> pd.DataFrame:
    df = dataset.copy()
    if coding is None:
        coding = DEFAULT_RESPONSE_CODING

    no_response_mask = pd.Series([False] * len(df), index=df.index)
    if 'refractory' in no_response_policy:
        refractory_mask = df[f"{combination} Ref."] == 1
        no_response_mask = no_response_mask | refractory_mask

    response_mask = pd.Series([False] * len(df), index=df.index)
    if 'exposed_non_refractory' in response_policy:
        exposed_non_refractory_mask = (df[f"{combination} Ref."] != 1) & (df[f"{combination} Exp."] == 1)
        response_mask = response_mask | exposed_non_refractory_mask
    if 'NDMM_SMM' in response_policy:
        # NDMM_mask = df['Disease Stage (MGUS=0, SMM=1, NDMM=2, RRMM=3, None=4)'].apply(lambda x: x in [1, 2])
        NDMM_mask = df['Disease'].apply(lambda x: x in ['NDMM', 'SMM'])
        response_mask = response_mask | NDMM_mask

    if raise_errors:
        assert sum(no_response_mask) > 0, f"no patients follow no response policy for combination: {combination}"
        assert sum(response_mask) > 0, f"no patients follow response policy for combination: {combination}"
        assert sum(response_mask & no_response_mask) == 0, \
            f"some patients follow both response and no response policies for combination: {combination}"

    response_series = pd.Series([coding["no_data"]] * len(df), index=df.index)
    response_series[response_mask] = coding["response"]
    response_series[no_response_mask] = coding["no_response"]

    df[f"{combination}_response"] = response_series

    return df


def load_and_process_clinical_data(clinical_data_path: Path, code_lower_case: bool, get_treatment_history: bool,
                                   get_hospital_stage: bool, get_post_treatment: bool,
                                   additional_cols: Optional[List[str]] = None,
                                   treatment_names: Optional[List[str]] = None):
    clinical_data = load_dataframe_from_file(clinical_data_path)
    if code_lower_case:
        clinical_data['Code'] = clinical_data['Code'].str.lower()

    requested_cols = []
    requested_cols += ["Code", "Biopsy sequence No."]
    if treatment_names is None:
        treatment_names = ["Bortezomib", "Ixazomib", "Carfilzomib", "Lenalidomide", "Thalidomide", "Pomalidomide",
                           "Cyclophosphamide", "Chemotherapy", "Venetoclax", "Dexamethasone", "Prednisone",
                           "Daratumumab", "Elotuzumab", "Belantamab", "Talquetamab", "Teclistamab", "Cevostamab",
                           "Selinexor", "Auto-SCT", "CART", "BiTE-BCMA"]

    if get_hospital_stage:
        hospital_stage_map = {0: 'NDMM',
                              1: "RRMM",
                              2: "SMM",
                              3: "MGUS",
                              4: "AL", 5: "AL",
                              6: "MGUS", 7: "MGUS",
                              8: None}
        hospital_stage = 'Plasma cell dyscrasia at Bx time(0=NDMM, 1=RRMM, 2=SMM 3=MGUS,4=NDAL, 5=RRAL, 6=NDSPC, 7=MGRS, 8=None)'
        clinical_data['Disease Stage Hospital'] = clinical_data[hospital_stage].map(hospital_stage_map)
        requested_cols += ['Disease Stage Hospital']
    if get_treatment_history:
        requested_cols += treatment_names
    if get_post_treatment:
        requested_cols += [f"{treatment}.2" for treatment in treatment_names]

    if additional_cols is not None:
        for col in additional_cols:
            if col not in clinical_data.columns:
                raise ValueError(f"{col} is requested but not in clinical data")
            else:
                requested_cols.append(col)
    requested_clinical_data = clinical_data[requested_cols]
    return requested_clinical_data


def add_response_columns_to_specific_treatment(dataset: pd.DataFrame, treatment: str,
                                               response_policy: str, no_response_policy: str,
                                               coding: Optional[dict] = None,
                                               raise_errors=True) -> pd.DataFrame:
    df = dataset.copy()
    if coding is None:
        coding = DEFAULT_RESPONSE_CODING

    # no_response_policy like 'pre_exposed|post_refractory'
    no_response_mask = pd.Series([False] * len(df), index=df.index)
    if 'pre_exposed' in no_response_policy:
        pre_exposed_mask = df[treatment] == 2 | df[treatment] == 1
        no_response_mask = no_response_mask | pre_exposed_mask
    elif 'pre_refractory' in no_response_policy:
        pre_refractory_mask = df[treatment] == 2
        no_response_mask = no_response_mask | pre_refractory_mask

    if 'last_line_exposed' in no_response_policy:
        last_line_exposed_mask = df[f"{treatment}.1"] == 2 | df[f"{treatment}.1"] == 1
        no_response_mask = no_response_mask | last_line_exposed_mask
    elif 'last_line_refractory' in no_response_policy:
        last_line_refractory_mask = df[f"{treatment}.1"] == 2
        no_response_mask = no_response_mask | last_line_refractory_mask

    if 'post_refractory' in no_response_policy:
        post_refractory_mask = df[f"{treatment}.2"] == 2
        no_response_mask = no_response_mask | post_refractory_mask

    # response_policy like 'NDMM_SMM|post_sensitive|not_exposed'
    response_mask = pd.Series([False] * len(df), index=df.index)
    if 'NDMM_SMM' in response_policy:
        # NDMM_mask = df['Disease Stage (MGUS=0, SMM=1, NDMM=2, RRMM=3, None=4)'].apply(lambda x: x in [1, 2])
        NDMM_mask = df['Disease'].apply(lambda x: x in ['NDMM', 'SMM'])
        response_mask = response_mask | NDMM_mask
    if 'post_sensitive' in response_policy:
        post_sensitive = dataset[f"{treatment}.2"] == 4
        treated_alone_mask = (~ dataset[dataset.columns[dataset.columns.str.contains("\.2")]].isna()).sum(axis=1) == 1
        response_mask = response_mask | (post_sensitive & treated_alone_mask)
    if 'not_exposed' in response_policy:
        not_exposed_mask = df[treatment].isna()
        response_mask = response_mask | not_exposed_mask

    if raise_errors:
        assert sum(no_response_mask) > 0, f"no patients follow no response policy for treatment: {treatment}"
        assert sum(response_mask) > 0, f"no patients follow response policy for treatment: {treatment}"

    response_series = pd.Series([coding["no_data"]] * len(df), index=df.index)
    response_series[response_mask] = coding["response"]
    response_series[no_response_mask] = coding["no_response"]

    df[f"{treatment}_response"] = response_series
    if sum(response_mask & no_response_mask) != 0:
        cols_to_print = ['Hospital.Code', 'Biopsy.Sequence', 'CD45', 'PC', 'Disease', 'Project', 'Cohort', 'Method',
                         f'{treatment}', f'{treatment}.2', f"{treatment}_response",
                         'Plasma cell dyscrasia at Bx time(0=NDMM, 1=RRMM, 2=SMM 3=MGUS,4=NDAL, 5=RRAL, 6=NDSPC, 7=MGRS, 8=None)']
        cols_to_print = list(set(cols_to_print).intersection(df.columns))
        warn = f"some patients follow both response and no response policies for treatment: {treatment}\n" \
               f"will consider them as non responders\n\n" \
               f"{df[response_mask & no_response_mask][cols_to_print]} "
        warnings.warn(warn)

    return df
