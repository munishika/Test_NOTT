#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 07:34:49 2024

@author: munishika
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

pValThres = 0.05
UseNormalDitribution = 0

# Simulate Group A
groupA_patientNB = 100;
groupA_patientId = np.random.randint(1,100000,size=groupA_patientNB)
if (UseNormalDitribution):
    groupA_AgeOnset = np.random.normal(60,5,groupA_patientNB)
else:
    groupA_AgeOnset = np.random.uniform(40,80,groupA_patientNB)    
groupA_sexAtBirth = np.random.choice(2,groupA_patientNB)
groupA_pd = pd.DataFrame({'PatientID':groupA_patientId, 'AgeOfOnset':groupA_AgeOnset, 'SexAtBirth':groupA_sexAtBirth})

# Simulate Group B
groupB_patientNB = 100;
groupB_patientId = np.random.randint(1,100000,size=groupA_patientNB)
if (UseNormalDitribution):
    groupB_AgeOnset = np.random.normal(65,4,groupA_patientNB)
else:
    groupB_AgeOnset = np.random.uniform(50,90,groupA_patientNB)
groupB_sexAtBirth = np.random.choice(2,groupA_patientNB)
groupB_pd = pd.DataFrame({'PatientID':groupB_patientId, 'AgeOfOnset':groupB_AgeOnset, 'SexAtBirth':groupB_sexAtBirth})
  

# Voilin plot to assess the distribution of Age of Onset
plt.violinplot(groupA_pd['AgeOfOnset'])
plt.violinplot(groupB_pd['AgeOfOnset'])

# test for differences of mean Age of Onset
#   1. test (here confirm) for normality of the data
res1=stats.shapiro(groupA_pd['AgeOfOnset'])
res2=stats.shapiro(groupB_pd['AgeOfOnset'])
# if both variables are normaly distributed
if ((res1.pvalue > pValThres) and (res2.pvalue > pValThres)):
    # Apply Welch's t-test
    print("Welch's t-test applied")
    res = stats.ttest_ind(groupA_pd['AgeOfOnset'], groupB_pd['AgeOfOnset'], equal_var=False)
else:
    # apply Mann Witney U test
    print("WMann Witney U test applied")
    res= stats.mannwhitneyu(groupA_pd['AgeOfOnset'], groupB_pd['AgeOfOnset']);

if (res.pvalue < pValThres):    
    print("There is a signficant difference (p=", res.pvalue, ") between Group A and Group B in terms of Age of onset")
else:
    print("There is no signficant differences (p=", res.pvalue, ") between Group A and Group B in terms of Age of onset")
