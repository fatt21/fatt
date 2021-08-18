import sys
from os import path
sys.path.append('.')
sys.path.append('..')
base_dir = path.dirname(path.realpath(__file__)) + '/'

import numpy as np
import pandas as pd
import Dataset


########################################################################
# Reads dataset
data_file = base_dir + 'dataset-raw.csv'
Dataset.getFromUrl('http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt', data_file)
column_names = [
    'community_name', 'state', 'country_code', 'community_code', 'fold', 'population', 'householdsize', 'racepctblack',
    'race_pct_white', 'race_pct_asian', 'race_pct_hisp', 'age_pct_12t21', 'age_pct_12t29', 'age_pct_16t24', 'age_pct_65up',
    'numb_urban', 'pct_urban', 'med_income', 'pct_w_wage', 'pct_w_farm_self', 'pct_w_inv_inc', 'pct_w_soc_sec', 'pct_w_pub_asst',
    'pct_w_retire', 'med_fam_inc', 'per_cap_inc', 'white_per_cap', 'black_per_cap', 'indian_per_cap', 'asian_per_cap',
    'other_per_cap', 'hisp_per_cap', 'num_under_pov', 'pct_pop_under_pov', 'pct_less9thgrade', 'pct_not_hs_grad', 'pct_bs_or_more',
    'pct_unemployed', 'pct_employ', 'pct_empl_manu', 'pct_empl_profserv', 'pct_occup_manu', 'pct_occup_mgmt_prof',
    'male_pct_divorce', 'male_pct_nev_marr', 'female_pct_div', 'total_pct_div', 'pers_per_fam', 'pct_fam2par', 'pct_kids2par',
    'pct_young_kids2par', 'pct_teen2par', 'pct_work_mom_young_kids', 'pct_work_mom', 'num_kids_born_never_mar',
    'pct_kids_born_never_mar', 'num_immig', 'pct_immig_recent', 'pct_immig_rec5', 'pct_immig_rec8', 'pct_immig_rec10',
    'pct_recent_immig', 'pct_rec_immig5', 'pct_rec_immig8', 'pct_rec_immig10', 'pct_speak_engl_only', 'pct_not_speak_engl_well',
    'pct_larg_house_fam', 'pct_larg_house_occup', 'pers_per_occup_hous', 'pers_per_own_occ_hous', 'pers_per_rent_occ_hous',
    'pct_pers_own_occup', 'pct_pers_dense_hous', 'pct_hous_less3br', 'med_numbr', 'hous_vacant', 'pct_hous_occup',
    'pct_hous_own_occ', 'pct_vacant_boarded', 'pct_vac_more6mos', 'med_yr_hous_built', 'pct_hous_no_phone', 'pct_wo_full_plumb',
    'own_occ_low_quart', 'own_occ_med_val', 'own_occ_hi_quart', 'own_occ_qrange', 'rent_low_q', 'rent_median', 'rent_high_q',
    'rent_qrange', 'med_rent', 'med_rent_pct_hous_inc', 'med_own_cost_pct_inc', 'med_own_cost_pct_inc_no_mtg', 'num_in_shelters',
    'num_street', 'pct_foreign_born', 'pct_born_same_state', 'pct_same_house85', 'pct_same_city85', 'pct_same_state85',
    'lemas_sworn_ft', 'lemas_swft_per_pop', 'lemas_swft_field_ops', 'lemas_swft_field_per_pop', 'lemas_total_req',
    'lemas_to_treq_per_pop', 'polic_req_per_offic', 'polic_per_pop', 'racial_match_comm_pol', 'pct_polic_white', 'pct_polic_black',
    'pct_polic_hisp', 'pct_polic_asian', 'pct_polic_minor', 'offic_assgn_drug_units', 'num_kinds_drugs_seiz',
    'polic_ave_ot_worked', 'land_area', 'pop_dens', 'pct_use_pub_trans', 'polic_cars', 'polic_operbudg',
    'lemas_pct_polic_on_patr', 'lemas_gang_unit_deploy', 'lemas_pct_offic_drug_un', 'polic_budg_per_pop', 'murders',
    'murd_per_pop', 'rapes', 'rapes_per_pop', 'robberies', 'robbb_per_pop', 'assaults', 'assault_per_pop', 'burglaries',
    'burgl_per_pop', 'larcenies', 'larc_per_pop', 'auto_theft', 'auto_theft_per_pop', 'arsons', 'arsons_per_pop',
    'violent_crimes_per_pop', 'non_viol_per_pop'
]
dataset = pd.read_csv(data_file, sep=',', header=None, names=column_names)


########################################################################
# Drops unused columns/rows
dataset.drop([
    'community_name', 'country_code', 'community_code', 'fold',
    'murders', 'murd_per_pop', 'rapes', 'rapes_per_pop', 'robberies', 'robbb_per_pop', 'assaults',
    'assault_per_pop', 'burglaries', 'burgl_per_pop', 'larcenies', 'larc_per_pop', 'auto_theft',
    'auto_theft_per_pop', 'arsons', 'arsons_per_pop', 'non_viol_per_pop'
], axis=1, inplace=True)
dataset.replace(to_replace='?', value=np.nan, inplace=True)
dataset.dropna(axis=0, subset=['violent_crimes_per_pop'], inplace=True)
dataset.dropna(axis=1, inplace=True)


########################################################################
# Preprocessing
binary = []
categorical = ['state']
numerical = Dataset.numericalColumns(dataset, binary, categorical, ['violent_crimes_per_pop'])

dataset = Dataset.normalizeColumnValues(dataset, categorical)
dataset = Dataset.standardize(dataset, numerical)
dataset = Dataset.oneHotEncoding(dataset, binary, categorical)
dataset = Dataset.binarizeByMedian(dataset, ['violent_crimes_per_pop'])
dataset = Dataset.selectLabelColumn(dataset, 'violent_crimes_per_pop')
training_set, test_set = Dataset.split(dataset, 0.8)


########################################################################
# Saves files
Dataset.save(dataset, base_dir + 'dataset.csv')
Dataset.save(training_set, base_dir + 'training-set.csv')
Dataset.save(test_set, base_dir + 'test-set.csv')
Dataset.exportColumns(dataset, base_dir + 'columns.csv')
