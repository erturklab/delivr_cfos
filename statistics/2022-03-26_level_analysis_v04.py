#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:12:48 2022

@author: wirrbel
"""
import pandas as pd
import numpy as np
import os

region_overview_file = '/home/wirrbel/2023-08-07_delivr_run_results/2023-08-15_delivr_run_results/re-calculated tables/region_overview.xlsx'
#alternatively, collapsed (will require addition of parent groups from uncollapsed file)
#region_overview_file = '/home/wirrbel/2023-08-07_delivr_run_results/2023-08-15_delivr_run_results/re-calculated tables/region_collapsed_overview.xlsx'

#load files 
collection_region_table = pd.read_excel(region_overview_file)
#collection_collapsed_table = pd.read_excel('/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/2022-03-21_stats/region_collapsed_overview.xlsx')

#load groups file (from assemble_overview_tables_submission_v03)
groups_excel_sheet = '/home/wirrbel/2023-08-07_delivr_run_results/2023-08-09_exp_groups_v02.xlsx'
groups = pd.read_excel(groups_excel_sheet)

#select the correct experiments
groups = groups[(groups["Exp"] == 'c26-1st') | (groups["Exp"] == 'c26-2nd')]
#groups = groups[(groups["Exp"] == 'c26-1st')]
#groups = groups[(groups["Exp"] == 'c26-2nd')]

#only take samples that have passed QC 
groups = groups.loc[groups['Check'] == 'OK']

def normalize_to_subgroup_average(df, groups, expname, groupname):
    #find all names in the subgroup
    subgrouplist = groups.loc[(groups["Exp"] == expname) & (groups["Group"] == groupname)]["Name"].to_list()
    #calculate control group mean 
    group_avg = df[subgrouplist].T.mean()
    
    #find all samples in this experiment
    exp_namelist = groups.loc[groups["Exp"] == expname]["Name"].to_list()
    #normalize all samples of this experiment to the control group (this includes the control group)
    df[exp_namelist] = df[exp_namelist].div(group_avg,axis=0)
       
    return df 

#extract experiments separately
c26_1st = collection_region_table[groups.loc[groups["Exp"] == 'c26-1st',"Name"].to_list()]
c26_2nd = collection_region_table[groups.loc[groups["Exp"] == 'c26-2nd',"Name"].to_list()]

#combine both experiments, then reattach the first couple columns 
collection_combined = pd.merge(c26_1st, c26_2nd, left_index=True, right_index=True)
#collection_combined = c26_1st.copy()
#collection_combined = c26_2nd.copy()
collection_combined = pd.merge(collection_region_table[collection_region_table.columns.to_list()[1:12]],collection_combined,left_index=True,right_index=True)

#files go here 
target_folder = '/home/wirrbel/2023-08-07_delivr_run_results/level_analysis_v02_with_tumorweight_kicked_out_all/'
try:
    os.makedirs(target_folder,exist_ok=True)
except:
    pass

#sort collection_region_table
cell_list = collection_combined.sort_values('structure-level',ascending=False).copy().fillna(0.0)

#replace background and root 
cell_list.loc[0,'parent_id'] = 0
cell_list.loc[1,'parent_id'] = 0
cell_list['parent_id'] = cell_list['parent_id'].astype(int)

#hardcoded sample list 
sample_list = cell_list.columns[11:].to_list()

#catalog structure levels in a dict 
#level_list = {}

for level_number in cell_list['structure-level'].unique():
    #debug
    print("current level_number: ",level_number)
    level = cell_list.loc[cell_list['structure-level'] == level_number]
    
    #save in level list dict 
    #level_list[level_number] = level
    
    #calculate the sum for each parent level present 
    sum_per_level = level.groupby('parent_id')[sample_list].sum()
       
    for parent, summed_level in sum_per_level.iterrows():
        
        #then, add to next level 
        cell_list.loc[cell_list['id'] == parent,sample_list] = cell_list.loc[cell_list['id'] == parent,sample_list]+(summed_level)
        
#quantify counting error 
#somehow seems to count 4000 - 9000 cells too many per brain 
overcount = cell_list.loc[cell_list['name'] == 'background',sample_list].squeeze() - collection_region_table[sample_list].sum()
print ('Here are the results. Positive numbers indicate overcounting \n',overcount)   

#optional: Normalize to control group
cell_list = normalize_to_subgroup_average(cell_list, groups, 'c26-1st', 'PBS')
cell_list = normalize_to_subgroup_average(cell_list, groups, 'c26-2nd', 'PBS')
 

#save to excel 
cell_list.to_excel(os.path.join(target_folder,'region_overview_level_collapsed.xlsx'))

### stats 
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

#create collection    
levelcollection = pd.DataFrame(columns=cell_list.columns)
#add mean columns
levelcollection = pd.concat([levelcollection,pd.DataFrame(columns=['c26_mean','nc26_mean','pbs_mean',])])

#add p-value columns
levelcollection = pd.concat([levelcollection,pd.DataFrame(columns=['p_c26_vs_nc26','p_nc26_vs_pbs','p_c26_vs_pbs',])])
#add corrected p-values columns
levelcollection = pd.concat([levelcollection,pd.DataFrame(columns=['pvals_corrected_c26_vs_nc26','pvals_corrected_nc26_vs_pbs','pvals_corrected_c26_vs_pbs'])])
                             
#cleanup before stats: convert all 0s to nans (to be dropped)
cell_list = cell_list.replace(0,np.nan)
#cleanup before stats: drop all nan rows
cell_list = cell_list.dropna(axis=0,how='any')

#stats
for level_number in cell_list['structure-level'].unique()[:-2]:
    
    #extract level
    level = cell_list.loc[cell_list['structure-level'] == level_number].copy()
    
    #remove zero-valued rows (empty rows, multiple testing correction wont work otherwise )
    #level = level.loc[(level.iloc[:,11:]!=0).min(axis=1)]

    #for uncollapsed region_overview
    #c26_reg = level.iloc[:,12:18].copy()
    #nc26_reg = level.iloc[:,18:24].copy()
    #pbs_reg = level.iloc[:,24:].copy()
    c26_reg = level.loc[:, groups[groups["Group"] == 'c26']['Name'].to_list()].copy()
    nc26_reg = level.loc[:, groups[groups["Group"] == 'nc26']['Name'].to_list()].copy()
    pbs_reg = level.loc[:, groups[groups["Group"] == 'PBS']['Name'].to_list()].copy()
  
    #generate all the tests 
    t_c26_vs_nc26,p_c26_vs_nc26 = stats.ttest_ind(c26_reg,nc26_reg,axis=1)
    t_nc26_vs_pbs,p_nc26_vs_pbs = stats.ttest_ind(nc26_reg,pbs_reg,axis=1)
    t_c26_vs_pbs,p_c26_vs_pbs = stats.ttest_ind(c26_reg,pbs_reg,axis=1)
    
    #### multiple correction #### 
    reject_1, pvals_corrected_c26_vs_nc26, alphaSidak_1, alphaBonferroni_1 = multipletests(p_c26_vs_nc26.astype('float'),alpha=0.1,method="fdr_bh")
    reject_2, pvals_corrected_nc26_vs_pbs, alphaSidak_2, alphaBonferroni_2 = multipletests(p_nc26_vs_pbs.astype('float'),alpha=0.1,method="fdr_bh")
    reject_3, pvals_corrected_c26_vs_pbs, alphaSidak_3, alphaBonferroni_3 = multipletests(p_c26_vs_pbs.astype('float'),alpha=0.1,method="fdr_bh")
    
    if True in reject_1:
        print('found a significant difference at level ', level_number, ' C26 vs NC26! \n, regions: ',level.loc[reject_1,'acronym'].values.tolist())
    if True in reject_2:
        print('found a significant difference at level ', level_number, ' NC26 vs PBS! \n, regions: ',level.loc[reject_2,'acronym'].values.tolist())
    if True in reject_3:
        print('found a significant difference at level ', level_number, ' C26 vs PBS! \n, regions: ',level.loc[reject_3,'acronym'].values.tolist())    

    #add averages per group 
    level['c26_mean'] = c26_reg.mean(axis=1)
    level['nc26_mean'] = nc26_reg.mean(axis=1)
    level['pbs_mean'] = pbs_reg.mean(axis=1)

    #add to level 
    level['p_c26_vs_nc26'] = p_c26_vs_nc26
    level['p_nc26_vs_pbs'] = p_nc26_vs_pbs
    level['p_c26_vs_pbs'] = p_c26_vs_pbs
    
    level['pvals_corrected_c26_vs_nc26'] = pvals_corrected_c26_vs_nc26
    level['pvals_corrected_nc26_vs_pbs'] = pvals_corrected_nc26_vs_pbs
    level['pvals_corrected_c26_vs_pbs'] = pvals_corrected_c26_vs_pbs
    
    
    #add to collection
    levelcollection = levelcollection.append(level)
    
    #save 
    #level.to_excel(os.path.join(target_folder,'region_overview_level_collapsed_stats_level_'+str(level_number)+'.xlsx'))
    
levelcollection.to_excel(os.path.join(target_folder,'region_overview_level_collapsed_stats_level_all.xlsx'))