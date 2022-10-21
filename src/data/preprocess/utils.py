import os
from typing import List, Dict

import pandas as pd
import torch
import numpy as np
from pandas.core.dtypes.concat import union_categoricals

from tqdm import tqdm
from definitions import EXTERNAL_DATA_DIR, DATA_DIR
from tqdm.auto import tqdm

SYMBOL_IDX = {
    "PAD": 0,
    "SEP": 1,
    "UNK": 2,
    "MASK": 3,
    "CLS": 4,
    # 'None': 5
}


def get_icd_omop():
    """
    Maps ICD to ICD OMOP IDs. Then, ICD OMOP IDs to SNOWMED OMOP ID standard terms.
    :return:
    """
    # TODO: refactor out
    mapping_path = '/home/vauvelle/pycharm-sftp/diffsurv/data/'
    vocab_dir = f"{mapping_path}/athena"
    vocab = {
        "concept": pd.read_csv(f"{vocab_dir}/CONCEPT.csv", sep='\t'),
        # "domain": pd.read_csv(f"{vocab_dir}/DOMAIN.csv", sep='\t'),
        # "class": pd.read_csv(f"{vocab_dir}/CONCEPT_CLASS.csv", sep='\t'),
        # "relationship": pd.read_csv(f"{vocab_dir}/RELATIONSHIP.csv", sep='\t'),
        # "drug_strength": pd.read_csv(f"{vocab_dir}/DRUG_STRENGTH.csv", sep='\t'),
        # "vocabulary": pd.read_csv(f"{vocab_dir}/VOCABULARY.csv", sep='\t'),
        # "concept_synonym": pd.read_csv(f"{vocab_dir}/CONCEPT_SYNONYM.csv", sep='\t'),
        # "concept_ancestor": pd.read_csv(f"{vocab_dir}/CONCEPT_ANCESTOR.csv", sep='\t'),
        "concept_relationship": pd.read_csv(f"{vocab_dir}/CONCEPT_RELATIONSHIP.csv", sep='\t')
    }
    #
    icd10_omop = vocab["concept"].query("vocabulary_id in ('ICD10')")[
        ["concept_code", "concept_id", "concept_name", 'vocabulary_id']].rename(columns={"concept_code": "code"})
    # icd10_omop = vocab["concept"][
    #     ["concept_code", "concept_id", "concept_name",'vocabulary_id']].rename(columns={"concept_code": "code"})
    icd10_omop["code"] = icd10_omop["code"].replace("\.", "", regex=True)
    icd10_omop["concept_id"] = icd10_omop["concept_id"].astype(str)

    domains = ["Condition", "Measurement", "Observation", "Procedure"]
    origin = "ICD10"
    target = 'SNOMED'
    origin_codes = vocab["concept"].query("vocabulary_id==@origin").query(
        "domain_id==@domains").concept_id.to_list()
    target_codes = vocab["concept"].query(
        "vocabulary_id==@target&(standard_concept=='S'|standard_concept=='C')").query(
        "domain_id==@domains").concept_id.to_list()
    snomed_map = vocab["concept_relationship"].query(
        "concept_id_1 == @origin_codes & concept_id_2 == @target_codes & relationship_id=='Maps to'")[
        ["concept_id_1", "concept_id_2"]] \
        .rename(columns={"concept_id_1": "concept_id_origin", "concept_id_2": "concept_id"}) \
        .merge(vocab["concept"][["concept_id", "concept_code", "concept_name", "domain_id", "concept_class_id",
                                 "vocabulary_id"]], on="concept_id", how="left") \
        .assign(concept_id=lambda x: x.concept_id.astype(str))

    icd10_omop = icd10_omop.rename(columns={"concept_id": "concept_id_icd"})
    icd10_omop.concept_id_icd = icd10_omop.concept_id_icd.astype(int)

    icd10_omop = icd10_omop.merge(snomed_map, left_on='concept_id_icd', right_on='concept_id_origin', how='left')
    icd10_omop = icd10_omop[icd10_omop.concept_id.notna()]

    # Ensure 1 to 1 mapping
    # Used to find order (largest keep)
    # icd10_omop[icd10_omop.duplicated(subset=['code'],keep=False)].concept_class_id.value_counts()
    icd10_omop.concept_class_id = icd10_omop.concept_class_id.astype('category').cat.reorder_categories(
        new_categories=['Observable Entity', 'Location', 'Physical Force', 'Social Context', 'Context-dependent',
                        'Procedure', 'Event', 'Clinical Finding'], ordered=True)
    icd10_omop = icd10_omop.sort_values(by='concept_class_id').drop_duplicates(subset=['code'],
                                                                               keep='last')  # 19109 -> 15776 shapes

    icd10_omop.rename(columns={'concept_name_y': 'concept_name'}, inplace=True)
    icd10_omop = icd10_omop.loc[:, ['code', 'concept_id', 'concept_name']]
    icd10_omop.loc[:, 'code_type_match'] = 'diag'
    return icd10_omop


def get_rxnorm_omop():
    """
    Maps ICD to ICD OMOP IDs. Then, ICD OMOP IDs to SNOWMED OMOP ID standard terms.
    :return:
    """
    # TODO: refactor out
    mapping_path = '/home/vauvelle/pycharm-sftp/diffsurv/data/'
    vocab_dir = f"{mapping_path}/athena"
    vocab = {
        "concept": pd.read_csv(f"{vocab_dir}/CONCEPT.csv", sep='\t'),
        # "domain": pd.read_csv(f"{vocab_dir}/DOMAIN.csv", sep='\t'),
        # "class": pd.read_csv(f"{vocab_dir}/CONCEPT_CLASS.csv", sep='\t'),
        # "relationship": pd.read_csv(f"{vocab_dir}/RELATIONSHIP.csv", sep='\t'),
        # "drug_strength": pd.read_csv(f"{vocab_dir}/DRUG_STRENGTH.csv", sep='\t'),
        # "vocabulary": pd.read_csv(f"{vocab_dir}/VOCABULARY.csv", sep='\t'),
        # "concept_synonym": pd.read_csv(f"{vocab_dir}/CONCEPT_SYNONYM.csv", sep='\t'),
        # "concept_ancestor": pd.read_csv(f"{vocab_dir}/CONCEPT_ANCESTOR.csv", sep='\t'),
        "concept_relationship": pd.read_csv(f"{vocab_dir}/CONCEPT_RELATIONSHIP.csv", sep='\t')
    }
    rx_omop = vocab["concept"].query("vocabulary_id=='RxNorm'")[
        ["concept_code", "concept_id", "concept_name", "domain_id", "concept_class_id",
         "vocabulary_id"]]  # .rename(columns={"concept_code": "code"})
    rx_omop = rx_omop.loc[:, ['concept_name', 'concept_code', 'concept_id']]
    rx_omop.rename(columns={'concept_code': 'code'}, inplace=True)
    rx_omop.loc[:, 'code_type_match'] = 'gp_scripts'

    return rx_omop


def get_opcs_omop():
    """
    Maps ICD to ICD OMOP IDs. Then, ICD OMOP IDs to SNOWMED OMOP ID standard terms.
    :return:
    """
    # TODO: refactor out
    mapping_path = '/home/vauvelle/pycharm-sftp/diffsurv/data/'
    vocab_dir = f"{mapping_path}/athena"
    vocab = {
        "concept": pd.read_csv(f"{vocab_dir}/CONCEPT.csv", sep='\t'),
        # "domain": pd.read_csv(f"{vocab_dir}/DOMAIN.csv", sep='\t'),
        # "class": pd.read_csv(f"{vocab_dir}/CONCEPT_CLASS.csv", sep='\t'),
        # "relationship": pd.read_csv(f"{vocab_dir}/RELATIONSHIP.csv", sep='\t'),
        # "drug_strength": pd.read_csv(f"{vocab_dir}/DRUG_STRENGTH.csv", sep='\t'),
        # "vocabulary": pd.read_csv(f"{vocab_dir}/VOCABULARY.csv", sep='\t'),
        # "concept_synonym": pd.read_csv(f"{vocab_dir}/CONCEPT_SYNONYM.csv", sep='\t'),
        # "concept_ancestor": pd.read_csv(f"{vocab_dir}/CONCEPT_ANCESTOR.csv", sep='\t'),
        "concept_relationship": pd.read_csv(f"{vocab_dir}/CONCEPT_RELATIONSHIP.csv", sep='\t')
    }
    #
    opcs_omop = vocab["concept"].query("vocabulary_id in ('OPCS4')")[
        ["concept_code", "concept_id", "concept_name", 'vocabulary_id']].rename(columns={"concept_code": "code"})
    # opcs_omop = vocab["concept"][
    #     ["concept_code", "concept_id", "concept_name",'vocabulary_id']].rename(columns={"concept_code": "code"})
    opcs_omop["code"] = opcs_omop["code"].replace("\.", "", regex=True)
    opcs_omop["concept_id"] = opcs_omop["concept_id"].astype(str)

    domains = ["Condition", "Measurement", "Observation", "Procedure"]
    origin = "OPCS4"
    target = 'SNOMED'
    origin_codes = vocab["concept"].query("vocabulary_id==@origin").query(
        "domain_id==@domains").concept_id.to_list()
    target_codes = vocab["concept"].query(
        "vocabulary_id==@target&(standard_concept=='S'|standard_concept=='C')").query(
        "domain_id==@domains").concept_id.to_list()
    snomed_map = vocab["concept_relationship"].query(
        "concept_id_1 == @origin_codes & concept_id_2 == @target_codes & relationship_id=='Maps to'")[
        ["concept_id_1", "concept_id_2"]] \
        .rename(columns={"concept_id_1": "concept_id_origin", "concept_id_2": "concept_id"}) \
        .merge(vocab["concept"][["concept_id", "concept_code", "concept_name", "domain_id", "concept_class_id",
                                 "vocabulary_id"]], on="concept_id", how="left") \
        .assign(concept_id=lambda x: x.concept_id.astype(str))

    opcs_omop = opcs_omop.rename(columns={"concept_id": "concept_id_icd"})
    opcs_omop.concept_id_icd = opcs_omop.concept_id_icd.astype(int)

    opcs_omop = opcs_omop.merge(snomed_map, left_on='concept_id_icd', right_on='concept_id_origin', how='left')
    opcs_omop = opcs_omop[opcs_omop.concept_id.notna()]

    # Ensure 1 to 1 mapping
    # Used to find order (largest keep)
    top_class = list(opcs_omop.concept_class_id.value_counts().index)
    opcs_omop.concept_class_id = opcs_omop.concept_class_id.astype('category').cat.reorder_categories(
        new_categories=top_class, ordered=True)
    opcs_omop = opcs_omop.sort_values(by='concept_class_id').drop_duplicates(subset=['code'],
                                                                             keep='first')  # 19109 -> 15776 shapes

    opcs_omop.rename(columns={"concept_name_y": "concept_name"}, inplace=True)
    opcs_omop = opcs_omop.loc[:, ['code', 'concept_id', 'concept_name']]
    opcs_omop.loc[:, 'code_type_match'] = 'oper'
    return opcs_omop


def get_phecode_map():
    icd10_phecode = os.path.join(EXTERNAL_DATA_DIR, 'icd10_phecodeX.csv')
    read2_phecode = os.path.join(EXTERNAL_DATA_DIR, 'read2_to_phecode.csv')
    readctv3_phecode = os.path.join(EXTERNAL_DATA_DIR, 'readctv3_to_phecode.csv')

    icd_phecode = pd.read_csv(icd10_phecode)
    icd_phecode.columns = ['code', 'description', 'phecode', 'phenotype', 'exclude_range_codes', 'exclude_range']
    icd_phecode.code = icd_phecode.code.str.replace('.', '', regex=False)
    icd_phecode.loc[:, 'code_type_match'] = 'diag'
    icd_phecode = icd_phecode.loc[:, ['code', 'code_type_match', 'phecode']]

    read2_phecode = pd.read_csv(read2_phecode)
    read2_phecode.columns = ['code', 'phecode']
    read2_phecode.loc[:, 'code_type_match'] = 'read_2'
    read2_phecode = read2_phecode.loc[:, ['code', 'code_type_match', 'phecode']]

    read3_phecode = pd.read_csv(readctv3_phecode)
    read3_phecode.columns = ['code', 'phecode']
    read3_phecode.loc[:, 'code_type_match'] = 'read_3'
    read3_phecode = read3_phecode.loc[:, ['code', 'code_type_match', 'phecode']]

    phecode_lookup = pd.concat([icd_phecode, read2_phecode, read3_phecode], axis=0)

    phecode_lookup.phecode = phecode_lookup.phecode.astype(str)
    phecode_lookup.phecode = phecode_lookup.phecode.str.replace('\.0', '', regex=False)

    # add new line to pandas dataframe

    return phecode_lookup


def get_read_omop():
    mapping_tables_dir = '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/external/Mapping Tables/'
    rc2_map = pd.read_csv(
        os.path.join(mapping_tables_dir, "Updated/Not Clinically Assured/rcmap_uk_20200401000001.txt"), sep='\t')
    rc3_map = pd.read_csv(
        os.path.join(mapping_tables_dir, "Updated/Clinically Assured/ctv3sctmap2_uk_20200401000001.txt"), sep='\t')

    rc2_map = rc2_map.query("MapStatus==1").drop_duplicates(subset=['ReadCode', 'ConceptId'])
    rc2_map.rename(columns={'ReadCode': 'code', 'ConceptId': 'concept_id'}, inplace=True)
    rc2_map = rc2_map.loc[:, ['code', 'concept_id']]
    rc2_map.loc[:, 'code_type_match'] = 'read_2'

    # Exclude term strings
    rc3_map = rc3_map.query("CTV3_TERMTYPE == 'P' and MAPSTATUS==1")
    rc3_map = rc3_map.sort_values(by=['IS_ASSURED'])
    rc3_map = rc3_map.drop_duplicates(subset=['CTV3_CONCEPTID', 'SCT_CONCEPTID'], keep='last')
    # rc3_map.query("IS_ASSURED == 0").shape 44389
    # rc3_map.query("IS_ASSURED == 1").shape 26385
    rc3_map.rename(columns={'CTV3_CONCEPTID': 'code', 'SCT_CONCEPTID': 'concept_id'}, inplace=True)
    rc3_map = rc3_map.loc[:, ['code', 'concept_id']]
    rc3_map.loc[:, 'code_type_match'] = 'read_3'

    rc_map = pd.concat([rc2_map, rc3_map], axis=0)

    # concept id to omop id...
    mapping_path = '/home/vauvelle/pycharm-sftp/diffsurv/data/'
    vocab_dir = f"{mapping_path}/athena"
    vocab_concept = pd.read_csv(f"{vocab_dir}/CONCEPT.csv", sep='\t')

    vocab_concept = vocab_concept.query("vocabulary_id == 'SNOMED'")
    vocab_concept.loc[:, 'concept_code'] = vocab_concept.concept_code.astype(int)

    rc_map = rc_map[rc_map.concept_id != '_DRUG']
    rc_map.concept_id = rc_map.concept_id.astype(int)
    rc_map = rc_map.merge(vocab_concept, left_on='concept_id', right_on='concept_code', how='left')

    rc_map = rc_map.loc[:, ['code', 'concept_id_y', 'concept_name', 'code_type_match']]
    rc_map.columns = ['code', 'concept_id', 'concept_name', 'code_type_match']

    return rc_map


def update_with_phecode_x(phe_data):
    """
    Update the phe_data with the phecode_x mapping
    :param phe_data:
    :return:
    """
    covariates = pd.read_parquet(os.path.join(DATA_DIR,'processed', 'covariates', 'eid_covariates.parquet'))
    phe_data = phe_data.merge(covariates.loc[:, ['eid', 'sex']], on='eid', how='left')

    sno_icd_phe_map = pd.read_feather(os.path.join(EXTERNAL_DATA_DIR, 'snomed_icd10_phecode_mapping_220407.feather'))
    sno_icd_phe_map.rename(columns={'phecode': 'phecode_X'}, inplace=True)
    sno_icd_phe_map.rename(columns={'mapRule': 'sex'}, inplace=True)
    sno_icd_phe_map.concept_id = sno_icd_phe_map.concept_id.astype(str).astype("category")

    c = union_categoricals([sno_icd_phe_map.concept_id, phe_data.concept_id])
    sno_icd_phe_map.concept_id = sno_icd_phe_map.concept_id.cat.set_categories(c.categories)
    phe_data.concept_id = phe_data.concept_id.cat.set_categories(c.categories)

    sno_icd_phe_map.sex = sno_icd_phe_map.sex.cat.rename_categories(
        {'IFA 248152002 | Female (finding) |': 0,
         'IFA 248153007 | Male (finding) |': 1,
         'OTHERWISE TRUE': 2,
         'TRUE': 3}
    )


    phe_data_x = phe_data.merge(sno_icd_phe_map.loc[:, ['concept_id',
                                                        'phecode_X', 'sex']],
                                on=['concept_id'], how='left')
    phe_data_x.sex_x = phe_data_x.sex_x.astype("category")
    c = union_categoricals([phe_data_x.sex_x.astype("category"), phe_data_x.sex_y])
    phe_data_x.sex_x = phe_data_x.sex_x.cat.set_categories(c.categories)
    phe_data_x.sex_y = phe_data_x.sex_y.cat.set_categories(c.categories)

    # remove rows with miss matching sex requirements and keep those with no requirements
    phe_data_x = phe_data_x.query("~((sex_y == 1 & sex_x == 0)|(sex_y == 0 & sex_x == 1))")
    phe_data_x.phecode_X = phe_data_x.phecode_X.cat.add_categories(["SEP", "UNK"])
    phe_data_x.loc[phe_data_x.phecode == "SEP", 'phecode_X'] = 'SEP'
    phe_data_x.phecode_X = phe_data_x.phecode_X.fillna('UNK')

    # death
    phe_data_x.phecode_X = phe_data_x.phecode_X.cat.add_categories(["death"])
    phe_data_x.code = phe_data_x.code.cat.add_categories(["death"])
    phe_data_x.code_type = phe_data_x.code_type.cat.add_categories(["death"])
    # phe_data_x.concept_id= phe_data_x.concept_id.cat.add_categories(["4306655"])
    deaths = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'application58356', 'death.txt'), delimiter='\t')
    deaths.loc[:, 'date'] = pd.to_datetime(deaths.date_of_death)
    deaths.sort_values(by=['date'], inplace=True)
    deaths.drop_duplicates(subset=['eid'], keep='last', inplace=True)  # around 60 are duplicated

    phe_data_x.drop(columns=['sex_y', 'sex_x'], inplace=True)

    # merge death data with phe data
    phe_data_x_deaths = phe_data_x.merge(deaths.loc[:, ['eid', 'date']], on=['eid', 'date'], how='inner')
    phe_data_x_deaths.code_type = 'death'
    phe_data_x_deaths.code = 'death'
    phe_data_x_deaths.phecode = 'death'
    phe_data_x_deaths.phecode_X = 'death'
    phe_data_x_deaths.concept_id = '4306655'
    phe_data_x_deaths.drop_duplicates(subset=['eid'], inplace=True)


    phe_data_x = pd.concat([phe_data_x, phe_data_x_deaths], axis=0)

    phe_data_x.sort_values(by=['eid', 'date'], inplace=True)
    phe_data_x.drop_duplicates(ignore_index=True, inplace=True)

    phe_data_x.to_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/omop/phe_data_x.csv', index=False)
    phe_data_x.to_feather('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/omop/phe_data_x.feather')
    return phe_data_x



def get_omop_map(save_path=os.path.join(EXTERNAL_DATA_DIR, 'omop_map.csv')):
    if os.path.exists(save_path):
        omop_map = pd.read_csv(save_path)
    else:
        opcs_omop = get_opcs_omop()
        read_omop = get_read_omop()
        icd10_omop = get_icd_omop()
        rxnorm_omop = get_rxnorm_omop()
        death = pd.DataFrame(
            [{'concept_name': 'death',
              'code': 'DEAD',
              'concept_id': '4306655',
              'code_type_match': 'death_nocause'}, ])

        omop_map = pd.concat([opcs_omop, read_omop, icd10_omop, rxnorm_omop, death], axis=0)

        omop_map.to_csv(save_path, index=False)

    return omop_map


def reformat_gnn_embedding(embedding):
    embedding.columns = ['concept_id', 'embedding']
    embedding.concept_id = embedding.concept_id.str.split('_').str[1]
    return embedding


# csvcut -c 'eid,31-0.0,34-0.0,53-0.0,54-0.0,21003-0.0' 58356.csv > /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/covariates.csv
def get_covariates(covariates_path=None):
    if os.path.exists(covariates_path):
        return pd.read_parquet(covariates_path)
    else:
        path = '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/covariates/covariates.csv'
        patient_base = pd.read_csv(path, low_memory=False, encoding='latin')
        patient_base.rename(columns={
            "31-0.0": "sex",
            "34-0.0": "yob",
            "52-0.0": "mob",
            "54-0.0": "center_ass",
            "53-0.0": "date_ass",
            "21003-0.0": "age_ass",
        }, inplace=True)
        # patient_base.loc[:, "dob"] = pd.to_datetime(patient_base.apply(
        #     lambda x: '{}/{}/01'.format(x.yob, x.mob) if x is not None else '{}/07/01'.format(x.yob), axis=1))
        patient_base.date_ass = pd.to_datetime(patient_base.date_ass)
        patient_base.loc[:, "year_ass"] = patient_base.date_ass.dt.year
        patient_base.to_parquet(os.path.join(DATA_DIR, 'processed', "covariates", "patient_base.parquet"))
        return patient_base


def align_pretrained_embedding(embedding, token2idx, save_path):
    token2idx_df = pd.DataFrame.from_records(list(token2idx.items()), columns=['concept_id', 'idx'])

    embedding_df = token2idx_df.merge(embedding, on='concept_id', how='left', validate='one_to_one')

    found_embeddings = embedding_df.dropna()
    # check
    missing = embedding_df[embedding_df.embedding.isna()]
    # omop_map = get_omop_map()
    # omop_map.query("concept_id in @missing.concept_id").code_type_match.value_counts()
    # missing = missing.loc[6:, :] # remove symbols
    # missing = missing[missing.concept_id != '_DRUG'] #
    # missing.concept_id = missing.concept_id.astype(int)

    # Init array of correct size with random 0-1
    rand_embeddings = np.random.rand(token2idx_df.shape[0], found_embeddings.embedding.iloc[0].shape[0])

    # Fill in found values
    rand_embeddings[found_embeddings.index] = np.concatenate(found_embeddings.embedding.values, axis=0).reshape(
        found_embeddings.shape[0], -1)

    embedding_tensor = torch.from_numpy(rand_embeddings)
    torch.save(embedding_tensor, save_path)
    return embedding_tensor, missing


def vocab_omop_embedding(embedding, token2idx,
                         save_path='/SAN/ihibiobank/denaxaslab/andre/ehrgraphs/models/embeddings/default.pkl') -> torch.Tensor:
    """
    Takes pretrained embbedding df and returns a tensor that matches vocab.
    Missing embeddings are random init
    :param embedding: dataframe with concept_id col and embedding col
    :param token2idx:
    :return:
    """
    assert list(embedding.columns) == ['concept_id', 'embedding']

    token2idx_df = pd.DataFrame.from_records(list(token2idx.items()), columns=['tokens', 'idx'])
    # Clean up
    token2idx_df.tokens = token2idx_df.tokens.str.upper()
    token2idx_df.tokens = token2idx_df.tokens.str.replace('\.', '', regex=True)

    omop_map = get_omop_map()
    omop_map.concept_id = omop_map.concept_id.astype(str)
    embedding.concept_id = embedding.concept_id.astype(str)

    # Get omop codes
    token2idx_df = token2idx_df.merge(omop_map.loc[:, ['code', 'concept_id']], left_on='tokens', right_on='code',
                                      how='left')
    # token2idx_df.concept_id.notna().sum()
    # Get embedding arrays
    token2idx_df = token2idx_df.merge(embedding, on='concept_id', how='left', validate="many_to_one")

    # Init tensor
    num_embeddings = token2idx_df.shape[0]
    embedding_dim = embedding.embedding.iloc[0].shape[0]
    embedding_tensor = torch.FloatTensor(num_embeddings, embedding_dim)
    embedding_tensor = torch.nn.init.normal_(embedding_tensor, mean=0, std=1)

    # Add in pretrained
    matched_embeddings = token2idx_df.embedding.dropna()
    pretrained_idxs = matched_embeddings.index.values
    embedding_tensor[pretrained_idxs] = torch.from_numpy(np.stack(matched_embeddings.values)).float()

    # embedding_tensor = torch.nn.Embedding.from_pretrained(embedding_tensor)
    if save_path is not None:
        torch.save(embedding_tensor, save_path)

    return embedding_tensor


def fit_vocab(data: List, min_count=None, min_proportion=None, top_n=None, label=False) -> Dict:
    """
    Fits a vocabulary to some data, returns as a dict
    :param label:
    :param top_n:
    :param data:
    :param min_count:
    :param min_proportion:
    :return:
    """
    counts = pd.Series(data).value_counts()
    # symbol_tokens = {padding_token, separator_token, unknown_token, mask_token, cls_token, 'None'}
    counts.drop(list(SYMBOL_IDX.values()), inplace=True,
                errors='ignore')
    proportions = counts / counts.sum()
    if min_count is not None:
        excluded_tokens = set(counts[counts < min_count].index)
        print(f'Excluding {counts[counts < min_count].sum()} tokens with count < {min_count}')
    elif min_proportion is not None:
        excluded_tokens = set(proportions[proportions < min_proportion].index)
        print(f'Excluding {proportions[proportions < min_proportion].sum()} tokens with proportion < {min_proportion}')
    elif top_n is not None:
        excluded_tokens = set(counts.iloc[top_n:].index)
        print(f'Excluding {counts.iloc[top_n:].sum()} tokens outside top n < {top_n}')
    else:
        excluded_tokens = set()

    data_tokens = set(data)
    data_tokens = data_tokens - {*SYMBOL_IDX.keys()} - excluded_tokens
    if label:
        unique_tokens = list(data_tokens)
    else:
        unique_tokens = list(SYMBOL_IDX.keys()) + list(data_tokens)
    idx2token = dict(enumerate(unique_tokens))
    token2idx = dict([(v, k) for k, v in idx2token.items()])
    vocab = {'idx2token': idx2token,
             'token2idx': token2idx}
    return vocab
