from lifelines import KaplanMeierFitter
import pandas as pd
import torch
from pdb import set_trace
import argparse
import matplotlib.pyplot as plt

# on Dell laptop (activate conda env 'pytorch_py3p10' and use 'python trainer.py')

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str,
                    default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/joint_fusion/',
                    help='Path to input data files')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

opt = parser.parse_args()

# create data mappings
# read the file containing gene expression and tile image locations for the TCGA-LUAD samples (mapped_data_16March)
mapping_df = pd.read_csv(opt.input_path + "mapping_df.csv")
print(mapping_df)

# set_trace()
################# prepare the data for the KM plot

# combine 'days_to_death' and 'days_to_last_followup' into a single column
# assuming that for rows where 'days_to_death' is NaN, 'days_to_last_followup' contains the censoring time
mapping_df['time'] = mapping_df['days_to_death'].fillna(mapping_df['days_to_last_followup'])

mapping_df['event_occurred'] = mapping_df['event_occurred'].map({'Dead': True, 'Alive': False})

kmf = KaplanMeierFitter()
kmf.fit(mapping_df['time'], event_observed=mapping_df['event_occurred'])


kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Estimate')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.show()
