import pandas as pd
from pathlib import Path
import scanpy as sc
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os
import random
import re
import shutil
from scipy.sparse import issparse
import glob
from typing import Optional, Union, Any
from anndata import AnnData
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
warnings.filterwarnings("ignore")


def preprocess_adata(adata, n_comps=25, n_neighbors=50):
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric='cosine')
    sc.tl.umap(adata, min_dist=0.1)
    return None
    
def preprocess_adata_subset_type(adata, cell_type, n_comps=25):
    adata_new = adata[adata.obs.cell_type == cell_type].copy()
    sc.pp.pca(adata_new, n_comps=n_comps)
    sc.pp.neighbors(adata_new, n_neighbors=50, metric='cosine')
    sc.tl.umap(adata_new, min_dist=0.1)
    return adata_new



def select_genes_by_variance(adata, var_threshold=0.1):
    """
    Selects genes based on variance threshold in sparse matrix and returns a DataFrame with gene names and their variances.

    Args:
    adata (AnnData): Annotated data matrix.
    var_threshold (float): Threshold for variance; default is 0.01.

    Returns:
    DataFrame: A DataFrame with columns 'gene' and 'variance' for genes with variance above the threshold.
    """
                                   
    if issparse(adata.X):
                                            
        mean = np.array(adata.X.mean(axis=0)).ravel()
        var = np.array(adata.X.power(2).mean(axis=0)).ravel() - mean**2
    else:
                                           
        var = np.var(adata.X, axis=0)

    adata.var['gene_variance'] = var

                                                  
    high_var_genes = adata.var['gene_variance'] > var_threshold

                                                               
    selected_genes_df = pd.DataFrame({
        'gene': adata.var.index[high_var_genes],
        'variance': adata.var['gene_variance'][high_var_genes]
    })
    print(f"Selected {len(selected_genes_df)} genes with variance greater than {var_threshold}.")

    return selected_genes_df

                       
def read_sciplex_adata(dpath:str,drug_to_smile,celltype:str=None):
    """Reads Sciplex raw data chunks and creates metadata

    Args:
        dpath (str): sciplex path
        celltype (str, optional): if celltype is provided, returns adata only for that celltype

    Returns:
        _type_: _description_
    """
    adatas = []
    for i in range(5):
        adatas.append(sc.read(Path(dpath)/f'sciplex_raw_chunk_{i}.h5ad'))
    adata = adatas[0].concatenate(adatas[1:])
    if isinstance(drug_to_smile,str):
        cols = ['Drug','SMILES','pathway']
        drug_to_smile = pd.read_csv(drug_to_smile,names=cols).set_index('Drug')['SMILES'].to_dict()
                          
        
        
    if celltype:
        adata = adata[adata.obs['cell_type'] == celltype].copy()
    
    adata.var['gene_id'] = adata.var.id.str.split('.').str[0]
                       
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
                       
    adata.obs['dose_val'] = adata.obs.dose.astype(float) / np.max(adata.obs.dose.astype(float))
    adata.obs.loc[adata.obs['product_name'].str.contains('Vehicle'), 'dose_val'] = 1.0
    adata.obs['dose_val'].value_counts()    
                                                                     
    adata.obs['product_name'] = [x.split(' ')[0] for x in adata.obs['product_name']]
    adata.obs.loc[adata.obs['product_name'].str.contains('Vehicle'), 'product_name'] = 'control'
    adata.obs['condition'] = adata.obs.product_name.copy()
    adata.obs['drug_dose_name'] = adata.obs.condition.astype(str) + '_' + adata.obs.dose_val.astype(str)
    adata.obs['cov_drug_dose_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.drug_dose_name.astype(str)
    adata.obs['cov_drug'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.condition.astype(str)
    adata.obs['control'] = [1 if x == 'control_1.0' else 0 for x in adata.obs.drug_dose_name.values]
    adata.obs["condition"] = adata.obs["condition"].astype('category')

    adata.obs["condition"] = adata.obs["condition"].cat.rename_categories({"(+)-JQ1": "JQ1"})
    adata.obs['SMILES'] = adata.obs.condition.map(drug_to_smile)
    adata[adata.obs["condition"] == "JQ1"].obs["SMILES"].unique()
    return adata

def process_row(row,adata_genes):
        genes = set(row.iloc[1:].dropna().tolist())
        if not genes.intersection(adata_genes):
            print(row.iloc[0])

                                            
            return random.sample(genes, 1)
        return []


def prepare_degs_adata(adata,
                    reaction_genes:str=None,
                    num_hvg:int=150):
    
    adata_genes = get_drug_degs(adata,
                               groupby='condition',
                               n_genes=30,
                               control_group='control')
    vt_genes = set(select_genes_by_variance(adata).gene.unique())
    print('variance threshold genes: ',len(vt_genes))
    
    if isinstance(reaction_genes,str):
        reaction_genes = pd.read_csv(reaction_genes)
        
    all_reaction_genes = pd.unique(reaction_genes[reaction_genes.columns[1:]].values.ravel())
    all_reaction_genes = all_reaction_genes[pd.notna(all_reaction_genes)]
                                                         
    vt_genes = vt_genes.intersection(all_reaction_genes)
                                     
                                                                                
                                                             
    adata_genes = adata_genes.union(all_reaction_genes)
                                                                           
                                                 
    
    reaction_genes = set(reaction_genes.apply(lambda row: process_row(row, adata_genes),
                                                          axis=1).explode().dropna())
    
    
    adata_fluxes_genes = adata_genes.union(reaction_genes)
                               
                                                                          
    get_cov_drug_dose_degs(adata, groupby='cov_drug',
                           covariate='cell_type',
                           control_group='control',
                           key_added='all_DEGs')
    
    cov_drug_dose_unique = adata.obs.cov_drug_dose_name.unique()
    remove_dose = lambda s: '_'.join(s.split('_')[:-1])
    cov_drug = pd.Series(cov_drug_dose_unique).apply(remove_dose)
    dose_no_dose_dict = dict(zip(cov_drug_dose_unique, cov_drug))
    uns_key = 'all_DEGs'
    df_DEGs = pd.Series(adata.uns[uns_key])
    new_DEGs_dict = {}
    for key, value in dose_no_dose_dict.items():
        if 'control' in key: continue
        new_DEGs_dict[key] = df_DEGs.loc[value]
    adata.uns[uns_key] = new_DEGs_dict
                 
    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvg, subset=False)
    hvgs = set(adata.var[adata.var['highly_variable']].index.tolist())
    adata_genes = hvgs.union(adata_fluxes_genes)
    adata = adata[:,adata.var.index.isin(adata_genes)].copy()
    return adata


def get_supermodule_info(project_dir, target_genes: list) -> pd.DataFrame:
    module_genes_df = pd.read_csv(project_dir/'fluxes/module_gene_m168.csv')
    modules_info_df = pd.read_csv(project_dir/'fluxes/Human_M168_information.symbols.csv')

    merged_df = module_genes_df.merge(modules_info_df, left_on=module_genes_df.columns[0], right_on='Unnamed: 0', how='left')

    supermodule_dict = {}
    for index, row in merged_df.iterrows():
        supermodule = 'SMID->' + str(row['Supermodule_id'])
        genes = row[1:len(module_genes_df.columns)].dropna().tolist()
        if supermodule in supermodule_dict:
            supermodule_dict[supermodule] = list(set(supermodule_dict[supermodule]) | set(genes))
        else:
            supermodule_dict[supermodule] = genes
            
    module_count = modules_info_df['Supermodule_id'].value_counts().to_dict()
    data = []
    for supermodule, genes in supermodule_dict.items():
        num_genes = len(genes)
        num_target_genes = len(set(genes) & set(target_genes))
        num_modules = module_count[int(supermodule.replace("SMID->", ""))]
        target_gene_percentage = round((num_target_genes / num_genes) * 100, 1)
        data.append([supermodule, num_genes, num_modules, num_target_genes, target_gene_percentage])


    df_results = pd.DataFrame(data, columns=['Supermodule', 'Number of Genes', 'Number of Modules', 'Number of Target Genes', 'Target Gene Match %'])
    return df_results


def get_cov_drug_dose_degs(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="all_DEGs",
    return_dict=False,
):

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
                        
                                                             
        control_group_cov = "_".join([cov_cat, control_group])

                                                                 
        adata_cov = adata[adata.obs[covariate] == cov_cat]

                      
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            method='t-test',
            n_genes=n_genes,
        )

                                                
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict


def get_drug_degs(
    adata,
    groupby,
    control_group,
    n_genes=50,
    rankby_abs=True)->set:
    gene_dict = {}
    control_group_cov = "_".join([control_group])
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        reference=control_group_cov,
        rankby_abs=rankby_abs,
        test = 't-test',
        n_genes=n_genes,
    )
    de_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
    for group in de_genes:
        gene_dict[group] = de_genes[group].tolist()

    adata_degs = set(genes for group in gene_dict.values() for genes in group)
    print(len(adata_degs))
    return set(adata_degs)

    
def build_save_path(base_path,
                    cell_line = None,
                    dataset_type:str = 'sciplex'):

    current_date_dir = datetime.utcnow().strftime('%Y-%m-%d')
    save_dir = os.path.join(base_path, cell_line if cell_line  else 'all',
                            current_date_dir,dataset_type)
    os.makedirs(save_dir, exist_ok=True)

                                                                       
    current_time_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    if cell_line:
        filename = f'sciplex_{cell_line}_{current_time_str}.h5ad'
    else:
        filename = f'sciplex_all_cell_lines_{current_time_str}.h5ad'
    
    return os.path.join(save_dir, filename)

def prepare_sciplex_splits(adata,drugs_split):
    
    if isinstance(drugs_split,str):
        drugs_split = pd.read_csv(drugs_split,index_col=0)
    drugs_split = {col: drugs_split[col].dropna().tolist() for col in drugs_split.columns if col}
    if 'split_ho_epigenetic_all' not in list(adata.obs):
                                                                  
        
        obs_train, obs_val = train_test_split(adata.obs.index, test_size=0.05)
                                                                      
        epigenetic_drugs = drugs_split['epigenetic_drugs']
        epigenetic_drugs_all = adata.obs.condition[adata.obs.pathway_level_1 == "Epigenetic regulation"].unique()
        adata.obs['split_ho_epigenetic_all'] = 'train'
        adata.obs.loc[adata.obs.index.isin(obs_val), 'split_ho_epigenetic_all'] = 'test'
        validation_cond = adata.obs.condition.isin(epigenetic_drugs_all[~epigenetic_drugs_all.isin(epigenetic_drugs)])
        val_idx = sc.pp.subsample(adata[validation_cond], 0.5, copy=True).obs.index
        adata.obs.loc[val_idx, 'split_ho_epigenetic_all'] = 'test'
        validation_cond = adata.obs.control.isin([True])
        val_idx = sc.pp.subsample(adata[validation_cond], 0.4, copy=True).obs.index
        adata.obs.loc[val_idx, 'split_ho_epigenetic_all'] = 'test'
        adata.obs.loc[adata.obs.condition.isin(epigenetic_drugs), 'split_ho_epigenetic_all'] = 'ood'
    return adata


def prepare_scflux_in(adata, reaction_genes, save_path, cell_line:str=None,slice_size=15000):
    """_summary_

    Args:
        adata: _description_
        reaction_genes: _description_
        save_path: _description_
        cell_line: _description_. Defaults to None.
        slice_size: _description_. Defaults to 40000.
    """
    
    if isinstance(reaction_genes,str):
        reaction_genes = pd.read_csv(reaction_genes)
        
    all_reaction_genes = pd.unique(reaction_genes[reaction_genes.columns[1:]].values.ravel())
    all_reaction_genes = all_reaction_genes[pd.notna(all_reaction_genes)]
    
    adata = adata.to_df().T
    adata = adata[adata.index.isin(reaction_genes)]
    num_observations = adata.shape[1]
    num_slices = num_observations // slice_size + (1 if num_observations % slice_size else 0)
    
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    save_dir = os.path.join(save_path,cell_line if cell_line  else 'all', current_date,'fluxes-in')
    if os.path.exists(save_dir): 
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    for i in range(num_slices):
        start_idx = i * slice_size
        end_idx = (i + 1) * slice_size if i != num_slices - 1 else adata.shape[1]
        subset_df = adata.iloc[:, start_idx:end_idx]
        
        file_name = f"fluxes-in_{i+1}_{current_date}.csv"
        subset_df.to_csv(os.path.join(save_dir, file_name))
                                                           
        
        

def prepare_scflux_in_with_control(adata, reaction_genes, save_path, cell_lines):
    """
    Prepares input data for scFEA, including control cells with each drug condition
    for the specified cell lines or all cell lines if not specified.
    """
    if isinstance(reaction_genes, str):
        reaction_genes = pd.read_csv(reaction_genes)
    all_reaction_genes = pd.unique(reaction_genes.iloc[:, 1:].values.ravel())
    all_reaction_genes = all_reaction_genes[pd.notna(all_reaction_genes)]
    filtered_adata = adata[:, adata.var_names.isin(all_reaction_genes)].copy()
    if cell_lines:
        filtered_adata = filtered_adata[filtered_adata.obs['cell_type'].isin(cell_lines)]
    else:
        cell_lines = filtered_adata.obs['cell_type'].unique()
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    save_dir = os.path.join(save_path, current_date, 'fluxes-in')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for cell_line in cell_lines:
        cell_line_data = filtered_adata[filtered_adata.obs['cell_type'] == cell_line]
        for condition in cell_line_data.obs['condition'].unique():
            if condition == 'control':
                continue
            condition_data = cell_line_data[cell_line_data.obs['condition'] == condition]
            control_data = cell_line_data[cell_line_data.obs['condition'] == 'control']
            combined_data = control_data.concatenate(condition_data)
            combined_df = combined_data.to_df().T
            file_name = f"{cell_line}_{condition}_fluxes-in_{current_date}.csv"
            combined_df.to_csv(os.path.join(save_dir, file_name))
            print(f"Saved file: {file_name}")


def prepare_scflux_by_cell_line(adata, reaction_genes, save_path, cell_lines):
    """
    Prepares input data for scFEA, including control cells with each drug condition
    for the specified cell lines or all cell lines if not specified.
    """
    if isinstance(reaction_genes, str):
        reaction_genes = pd.read_csv(reaction_genes)
    all_reaction_genes = pd.unique(reaction_genes.iloc[:, 1:].values.ravel())
    all_reaction_genes = all_reaction_genes[pd.notna(all_reaction_genes)]
    filtered_adata = adata[:, adata.var_names.isin(all_reaction_genes)].copy()
    if cell_lines:
        filtered_adata = filtered_adata[filtered_adata.obs['cell_type'].isin(cell_lines)]
    else:
        cell_lines = filtered_adata.obs['cell_type'].unique()
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    save_dir = os.path.join(save_path, current_date, 'fluxes-in')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print('cell lines',cell_lines)
    for cell_line in cell_lines:
        cell_line_data = filtered_adata[filtered_adata.obs['cell_type'] == cell_line]
        combined_df = cell_line_data.to_df().T
        file_name = f"{cell_line}_fluxes-in_{current_date}.csv"
        combined_df.to_csv(os.path.join(save_dir, file_name))
        print(f"Saved file: {file_name}")



def combine_scflux_out(adata_path, fluxes_dir):
    """
    Args:
        adata_path:
        fluxes_dir:
        
    Returns:
        combined fluxes from sc-flux.
    """
                                         
    all_files = os.listdir(fluxes_dir)
    pattern = re.compile(r'flux_fluxes-in_(\d+)_2023-10-01.csv')
    sorted_files = sorted(
        (file for file in all_files if pattern.match(file)),
        key=lambda x: int(pattern.match(x).group(1))
    )    
    
    print(sorted_files)
    flux_dfs = [pd.read_csv(os.path.join(fluxes_dir, file)) for file in sorted_files]
    combined_flux_df = pd.concat(flux_dfs)

                                
    adata = sc.read_h5ad(adata_path)
    adata_indices = adata.obs.index.tolist()
    combined_flux_df = combined_flux_df.set_index(combined_flux_df.columns[0])
    
                                                                
    combined_flux_df = combined_flux_df.reindex(adata_indices)
                                  
    combined_flux_df.to_csv(os.path.join(fluxes_dir, 'combined_fluxes.csv'))
    print(combined_flux_df)
    return combined_flux_df



def scale_fluxes_separately(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales each flux column in the given DataFrame to the range [-1, 1] separately.

    """

    scaled_df = df.copy()
    
    flux_columns = scaled_df.columns
    
    for col in flux_columns:
        col_min = scaled_df[col].min()
        col_max = scaled_df[col].max()
        if col_max == col_min:
            scaled_df[col] = 0
        else:
            scaled_df[col] = 2 * (scaled_df[col] - col_min) / (col_max - col_min) - 1
    
    return scaled_df

def marson_preprocess(
    adata: Union[AnnData, str],
    reaction_genes: Union[str, Any],
    save_dir: str,
    min_genes: Optional[int] =750,
    mt_frac: Optional[int] = 0.2,
    n_top_genes: Optional[int] = 1500,
    plot: Optional[bool] = True
    ):
    if isinstance(adata, str):
        adata = sc.read_h5ad(adata)
    if isinstance(reaction_genes, str):
        reaction_genes = pd.read_csv(reaction_genes)
        
    all_reaction_genes = pd.unique(reaction_genes.iloc[:, 1:].values.ravel())
    all_reaction_genes = all_reaction_genes[pd.notna(all_reaction_genes)]
    
    adata.var['gene_id'] = adata.var.name.str.split('.').str[0]
    adata.var_names = adata.var.name.astype(str).values
    adata.var_names_make_unique()
    adata.obs = adata.obs.rename(columns={'cluser_name': 'cluster_name'})
    adata = adata[adata.obs['nCount_RNA'] > 0]
    adata = adata[adata.obs['donor'] != 'Unassigned']
    
    print(f"total unique cell lines :{adata.obs.celltype.unique()}")
                                         
    adata.obs['n_counts'] = np.ravel(adata.X.sum(1))
                                                                      
    adata.obs['n_genes'] = np.ravel(np.sum(adata.X > 0, axis=1))
                                                
    adata.var['mito'] = adata.var_names.str.contains("MT-")
                                                                               
    adata.obs['mt_frac'] = np.ravel(adata.X[:, adata.var.mito].sum(1)) / adata.obs['n_counts'].values

    adata = adata[adata.obs['n_genes'] > min_genes]
    adata = adata[adata.obs['mt_frac'] < mt_frac]
    
                                                                              
    print(f"verifying count data: {adata.X.max()}")
    sc.pp.normalize_total(adata)
                                            
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger')
    hvgs = adata[:,adata.var.highly_variable].var_names.unique().tolist()
    scfea_common = set(adata.var.gene_id.unique()).intersection(all_reaction_genes)
    targeted_genes = set(adata.obs.target.values.unique())
    selected_genes = set(hvgs).union(targeted_genes)

    print(f"total selected genes: {len(selected_genes)}")

    adata = adata[:, (adata.var_names.isin(selected_genes) | (adata.var.gene_id.isin(scfea_common)))]
                                                         
    adata = adata[adata.obs['donor'].str.contains('Donor')]
    adata = adata[adata.obs['celltype'].str.contains('CD4|CD8')]
    adata = adata[adata.obs['stim_celltype'].str.contains('cd4|cd8')]
    adata.obs['control'] = [1 if x == 'NTC' else 0 for x in adata.obs['target'].values]
    adata.obs['dose'] = 1.
    
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    if plot:
        adata.obs['stim_celltype'] = adata.obs['stim_celltype'].str.replace('_', ' ').str.title()
        sc.pl.umap(
            adata, 
            color='stim_celltype',
            size=70,
            title='Projection of Stimulated Cell Types',
            frameon=False,                                        
            legend_loc='on data',
            legend_fontsize=12,
            alpha=0.8,  
            edgecolor='k',  
            linewidth=0.1)
    adata.obs['stim_celltype'] = adata.obs['stim_celltype'].str.replace(' ', '_').str.lower()
    ood_genes = get_marson_ood_genes(adata)
    print("ood_genes are",ood_genes)
    adata.obs['cov_pert'] = adata.obs['celltype'].astype(str) + "_" + adata.obs['target'].astype(str)
    adata.obs['cov_pert_dose'] = adata.obs['celltype'].astype(str) + "_" + adata.obs['target'].astype(str) + "_" +\
                                adata.obs['dose'].astype(str)
    get_cov_drug_dose_degs(adata,groupby='cov_pert',
                            control_group='NTC',
                            covariate='celltype',
                            key_added='all_DEGs')
    
    adata.obs['condition'] = adata.obs['target']                     
    adata.obs['split'] = 'NA'
    adata.obs.loc[(adata.obs['stim_celltype'] == 'restimulated_cd8')\
        & (adata.obs['target'].isin(ood_genes)), 'split'] = 'ood'
    idx = np.where(adata.obs['split']=='NA')[0]
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    adata.obs.iloc[idx_train, adata.obs.columns.get_loc('split')] = 'train'
    adata.obs.iloc[idx_test, adata.obs.columns.get_loc('split')] = 'test'
    print(f"splits: {adata.obs.split.value_counts()}")
    
    os.makedirs(save_dir,exist_ok=True)
    adata.write_h5ad(os.path.join(save_dir, 'marson_cleaned_fate.h5ad'))
    
def get_marson_ood_genes(adata: AnnData):
    results = []
    for cond1 in adata.obs.target.unique():
        ad1 = adata[adata.obs.target == cond1]
        ad2 = adata[adata.obs.target != cond1]
        mean1 = ad1.X.mean(0)
        mean2 = ad2.X.mean(0)
        l2 = np.linalg.norm(mean1-mean2)
        results.append({
            'cond1': cond1,
            'L2': l2
        })
    df_vs_rest = pd.DataFrame(results)
    print(df_vs_rest)
    df_vs_rest.sort_values(by='L2').tail(20)
    ood_genes = df_vs_rest.sort_values(by='L2').tail(20).cond1.values
    return ood_genes


    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the data and save.")
        
                                                       
    parser.add_argument("--scfea_module_gene_path", default= Path("data/scfea/module_gene_m168.csv"), help="Path to module genes CSV.")
    parser.add_argument("--dataset_name", default=None, help="Name of the dataset e.g. Marson, Sciplex, Norman.")
    parser.add_argument("--sc_adata_path", default=None, help="single cell adata file path")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    adata = read_sciplex_adata(dpath=args.dpath, drug_to_smile=args.drug_to_smiles,celltype=args.cell_line)
    adata = prepare_degs_adata(adata, args.fgenes)
    adata = prepare_sciplex_splits(adata, args.sciplex_splits)
    save_path = build_save_path(args.base_path,args.cell_line)
    adata.write(save_path)
    prepare_scflux_in(adata, args.fgenes,args.base_path,args.cell_line)
    
    print(f"Data saved to: {save_path}")
    
                                                
                                         