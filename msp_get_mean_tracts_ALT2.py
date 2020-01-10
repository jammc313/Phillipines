#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import msprime as msp
import random
import numpy as np
import scipy
from scipy import stats
import math
import pandas
import itertools
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
import allel; print('scikit-allel', allel.__version__)
from allel.stats.misc import jackknife


# ## MSPRIME SIMULATIONS

# This run, recombination = 1.25 e-8 (equal to mu)
# This run has 1 run with no replicate simulations, L=100M (chr sized sample).
# 10 samples per population. The full haplotype matrix is returned.
# Simulates the coalescent with recombination under the specified model parameters and returns the resulting :class:`tskit.TreeSequence`. Note that Ne is the effective diploid population size (so the effective number of genomes in the population is 2*Ne), but ``sample_size`` is the number of (monoploid) genomes sampled.
# 
# Source pops related by ...
# nS vector of sample sizes (monoploid genomes) in each S, i.e. (nS1, nS2, nS3)
# tS vector ages of samples in each S, i.e. (tS1, tS2, tS3)
# t12 time of S1,S2 split (first split)
# t123 time of ((S1, S2), S3) split
# ta vector of admixture times from S2 and S3 to S1
# f vector of admixture fractions from from S2 and S3 to S1
# N vector of effective sizes in order of population numbers
# L is length of simulated region in bases (set to 200MB here)
# 
# Here we are looking at admixed segment length distributions, D tests, and % shared archaic SNP blocks between Ayta and Papuan for Neanderthal and Denisovan. The model we are testing is the Alt2, with an admixture event from Neanderthal into the shared ancestor of all Non-Africans, a Denisovan admixture event into Papuans, and a separate Denisovan admixture event into Ayta.

# In[ ]:


# enter functions

DEN0, DEN1, DEN2, DEN3, AFR, CEU, PAP, AYT, NEA, CHM = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9


def set_up_pops(nS, tS):
    #historical series within pop S1
    samples = [msp.Sample(population=DEN0, time=tS[0])]*(1*nS[0]) # Denisovan 0
    samples.extend([msp.Sample(population=DEN1, time=tS[0])]*(1*nS[1])) # Denisovan 1
    samples.extend([msp.Sample(population=DEN2, time=tS[0])]*(1*nS[2])) # Denisovan 2
    samples.extend([msp.Sample(population=DEN3, time=tS[1])]*(1*nS[3])) # Denisovan 3 (Altai)
    samples.extend([msp.Sample(population=AFR, time=tS[0])]*(1*nS[4])) # Africa
    samples.extend([msp.Sample(population=CEU, time=tS[0])]*(1*nS[5])) # European
    samples.extend([msp.Sample(population=PAP, time=tS[0])]*(1*nS[6])) # Papuan
    samples.extend([msp.Sample(population=AYT, time=tS[0])]*(1*nS[7])) # Negrito (Ayta)
    samples.extend([msp.Sample(population=NEA, time=tS[2])]*(1*nS[8])) # Neanderthal
    samples.extend([msp.Sample(population=CHM, time=tS[0])]*(1*nS[9])) # Chimp
    return samples


def set_up_demography(t67, t75, t54, t10, t20, t03, t83, t34, t49, ta1, ta2, ta3, f):
    #divergence of source populations (topology of tree)
    source_divergence = [msp.MassMigration(time=t67, source=PAP, destination=AYT, proportion=1),  
                        msp.MassMigration(time=t75, source=AYT, destination=CEU, proportion=1), 
                        msp.MassMigration(time=t54, source=CEU, destination=AFR, proportion=1), 
                        msp.MassMigration(time=t10, source=DEN1, destination=DEN0, proportion=1), 
                        msp.MassMigration(time=t20, source=DEN2, destination=DEN0, proportion=1),
                        msp.MassMigration(time=t03, source=DEN0, destination=DEN3, proportion=1),
                        msp.MassMigration(time=t83, source=NEA, destination=DEN3, proportion=1),
                        msp.MassMigration(time=t34, source=DEN3, destination=AFR, proportion=1),
                        msp.MassMigration(time=t49, source=AFR, destination=CHM, proportion=1)] 
    #admixture times and proportions
    admix = [msp.MassMigration(time=ta1, source=AYT, destination=DEN2, proportion=f[0]),
            msp.MassMigration(time=ta2, source=PAP, destination=DEN1, proportion=f[1]), #fraction from Denisovan 1 to Negrito/Papuan
            msp.MassMigration(time=ta3, source=CEU, destination=NEA, proportion=f[2])] #fraction from Neanderthal to OoA pops
    #population parameter changes
    N_change = [msp.PopulationParametersChange(time=2400, initial_size=2000, growth_rate=0, population_id=CEU),
               msp.PopulationParametersChange(time=2800, initial_size=5000, growth_rate=0, population_id=CEU)]
    #combine and sort the demography
    demography = source_divergence + admix + N_change
    return sorted(demography, key = lambda x: x.time)


# Enter data
L=100000000
mu=1.25e-8
r=1.25e-8

generation_time=25
T67=46000/generation_time  # PAP joins AYT  
T75=55000/generation_time  # AYT joins CEU 
T54=70000/generation_time  # CEU joins AFR
T10=200000/generation_time # Denisovan 1 joins Denisovan 0
T20=200000/generation_time # Denisovan 2 joins Denisovan 0 
T03=300000/generation_time # Denisovan 0 joins Denisovan 3 (Altai) 
T83=400000/generation_time # Neanderthal joins Denisovan 3 (Altai) 
T34=600000/generation_time # Denisovan 3 (Altai) joins AFR
T49=4000000/generation_time # AFR joins Chimp
 
TA1=35000/generation_time   
TA2=45000/generation_time   
TA3=68000/generation_time   

TS_NEA=60000/generation_time
TS_DEN3=40000/generation_time

NumSamples=100
nS=[10]*10
tS=[0,TS_DEN3,TS_NEA]
f=[0.06, 0.04, 0.02]
N=[1500,1500,1500,1500,15000,5000,3500,3500,2000,30000]
seed=None

samples = set_up_pops(nS,tS)
demography = set_up_demography(T67, T75, T54, T10, T20, T03, T83, T34, T49, TA1, TA2, TA3, f)
pops = [msp.PopulationConfiguration(initial_size = n) for n in N]

# Use demography debugger for sanity check that you are simulating correct demography
#dd = msp.DemographyDebugger(
#    population_configurations=pops,
#    demographic_events=demography)
#dd.print_history()


# In[ ]:


# create empty array to take results
res_arr = np.zeros([10,7])


# In[ ]:

# some functions

# get records of all genuine migrating tracts
def get_migrating_tracts(ts):
    migrating_tracts_DEN1, migrating_tracts_DEN2, migrating_tracts_NEA = [], [], []
    # Get all tracts that migrated into the archaic populations
    for migration in ts.migrations():
        if migration.dest == DEN1:
            migrating_tracts_DEN1.append((migration.left, migration.right))
        elif migration.dest == DEN2:
            migrating_tracts_DEN2.append((migration.left, migration.right))
        elif migration.dest == NEA:
            migrating_tracts_NEA.append((migration.left, migration.right))
    return np.array(migrating_tracts_DEN1), np.array(migrating_tracts_DEN2), np.array(migrating_tracts_NEA) 

# In[ ]:

# run simulations

numRuns=10
for aRun in range(numRuns):

    sims = msp.simulate(samples=samples, Ne=N[0], population_configurations=pops, demographic_events=demography, mutation_rate=mu, length=L, recombination_rate=r, record_migrations=True, random_seed=seed)

    mig_DEN1, mig_DEN2, mig_NEA = get_migrating_tracts(sims)
    mean_tract_den1 = np.mean(mig_DEN1[:,1] - mig_DEN1[:,0], axis=0).astype(int)
    mean_tract_den2 = np.mean(mig_DEN2[:,1] - mig_DEN2[:,0], axis=0).astype(int)
    mean_tract_nea = np.mean(mig_NEA[:,1] - mig_NEA[:,0], axis=0).astype(int)

    ## Comparing putatively assigned "SNP tracts"

    # function to get copy of dataset
    def get_dataset(sim_mat):
        mat_full = np.zeros((sim_mat.shape[0],sim_mat.shape[1]+2))
        mat_full[:, 2:] = sim_mat
        return mat_full

    geno_mat_info = get_dataset(sims.genotype_matrix())

    for variant in sims.variants():
        geno_mat_info[variant.site.id,0]=variant.site.id+1
        geno_mat_info[variant.site.id,1]=variant.site.position


    # free up some RAM
    del sims

    # function to downsample full haplotype array to ordered and sorted panel of 0.25M SNPs (will lead to 250 blocks)

    def downsample_sort(big_arr, size):
        all_anc_index = big_arr[:,42:82].sum(axis=1)!=0     # index SNPs ancestral in all modern pop samples
        sub_arr1 = big_arr[all_anc_index,:]
        all_der_index = sub_arr1[:,42:82].sum(axis=1)!=40    # index SNPs derived in all modern pop samples
        sub_arr2 = sub_arr1[all_der_index,:]
        sub_arr3 = sub_arr2[np.random.choice(sub_arr2.shape[0], size, replace=False)]    # Downsample to panel of given SNPs
        sort_arr = sub_arr3[sub_arr3[:,0].argsort()] # sort based on first column (SNP index)
        return sort_arr

    geno_mat_sub_ord = downsample_sort(geno_mat_info.astype(int), 250000)
    geno_mat_sub_ord[:,0] = np.arange(1,250001)


    # Function uses all 10 samples per pop to get allele counts for estimating D stats
    # input data need to be of scikit.allel form 'AlleleCountsArray' for each population, with each SNP represented by number of anc & der alleles
    # function to take haplotype array and count alleles in each population sample of 5 diploids: 

    def pop_sample_ac(geno_mat):
        haplo_arr = allel.HaplotypeArray(geno_mat)
        ac_one = haplo_arr[:,0:10].count_alleles()
        ac_two = haplo_arr[:,10:20].count_alleles()
        ac_three = haplo_arr[:,20:30].count_alleles()
        ac_four = haplo_arr[:,30:40].count_alleles()
        ac_five = haplo_arr[:,40:50].count_alleles()
        ac_six = haplo_arr[:,50:60].count_alleles()
        ac_seven = haplo_arr[:,60:70].count_alleles()
        ac_eight = haplo_arr[:,70:80].count_alleles()
        ac_nine = haplo_arr[:,80:90].count_alleles()
        ac_ten = haplo_arr[:,90:100].count_alleles()
        # stack arrays with frames = population allele counts for all SNPs
        arrays = [ac_one,ac_two,ac_three,ac_four,ac_five,ac_six,ac_seven,ac_eight,ac_nine,ac_ten]
        ac_All = np.stack(arrays, axis=0)
        return ac_All

    ac_AllSamples = pop_sample_ac(geno_mat_sub_ord[:, 2:].astype(int))

    # make index array of relevant populations for different D test comparisons (D{AFR,DEN3;AYT/PAP,CEU})
    D_test_index = [[4,3,5,7],[4,3,5,6]]   

    # array to take D test results output for all above comparisons, returns d,se,z
    D_mean_arr = np.array([(allel.average_patterson_d(ac_AllSamples[x[0]], ac_AllSamples[x[1]], ac_AllSamples[x[2]], ac_AllSamples[x[3]], 1000)[0:3]) for x in D_test_index])

    # Divide the allele count array up into 1000 SNPs = 250 blocks.
    # perform mean patterson D test (jackknife=100) on given populations for each block

    def get_D_test_blocks(ac_arr,pop1,pop2,pop3,pop4):
        arr_tp1 = np.transpose(ac_arr, (1,0,2))    # order SNPS,INDS,AC
        arr_rshp = arr_tp1.reshape(250,1000,10,2)    # reshape with 250 blocks
        arr_tp2 = np.transpose(arr_rshp, (0,2,1,3))    # transpose again in preparation for D tests
        mean_D_blocks = np.array([(allel.average_patterson_d(block[pop1], block[pop2], block[pop3], block[pop4], 100)[0:3]) for block in arr_tp2])
        return mean_D_blocks

    # For each block returns: meanD, se, z

    ## ASSIGNING PUTATIVE DENISOVAN BLOCKS

    # First test for Denisovan ancestry in each block for Ayta
    D1_res_AYT = get_D_test_blocks(ac_AllSamples,AFR,DEN3,CEU,AYT)

    # Now test for Denisovan ancestry in each block for Papuan
    D1_res_PAP = get_D_test_blocks(ac_AllSamples,AFR,DEN3,CEU,PAP)

    # Now the second D test. If a block is positive from this and overlaps with a > 3 z score in previous D test,
    # we assign Denisovan ancestry
    D2_res_AYT = get_D_test_blocks(ac_AllSamples,AFR,AYT,NEA,DEN3)

    # Do the same for Papuan
    D2_res_PAP = get_D_test_blocks(ac_AllSamples,AFR,PAP,NEA,DEN3)


    def get_DEN_blocks(d1_PAP, d1_AYT, d2_PAP, d2_AYT):
        put_DEN_PAP = (d1_PAP[:,2] >= 3) & (d2_PAP[:,0] > 0)
        put_DEN_AYT = (d1_AYT[:,2] >= 3) & (d2_AYT[:,0] > 0)
        DEN_blocks_PAP = np.sum(put_DEN_PAP)
        DEN_blocks_AYT = np.sum(put_DEN_AYT)
        total_DEN_blocks = DEN_blocks_PAP + DEN_blocks_AYT
        shared_DEN_blocks = np.sum((put_DEN_PAP == True) & (put_DEN_AYT == True)) / total_DEN_blocks * 100
        return shared_DEN_blocks, DEN_blocks_PAP, DEN_blocks_AYT

    shared_DEN_PAPvsAYT, putative_DEN_PAP, putative_DEN_AYT = get_DEN_blocks(D1_res_PAP, D1_res_AYT, D2_res_PAP, D2_res_AYT)

    ## ASSIGNING PUTATIVE NEANDERTHAL BLOCKS

    # First test for Neanderthal ancestry in each block in Ayta
    D3_res_AYT = get_D_test_blocks(ac_AllSamples,CHM,NEA,AFR,AYT)

    # First test for Neanderthal ancestry in each block in Papuan
    D3_res_PAP = get_D_test_blocks(ac_AllSamples,CHM,NEA,AFR,PAP)

    # Do the 2nd D test. If a block is negative from this & overlaps with a > 3 z score in last test, assign Neanderthal
    D4_res_AYT = get_D_test_blocks(ac_AllSamples,AFR,AYT,NEA,DEN3)

    # The same for Papuan. 
    D4_res_PAP = get_D_test_blocks(ac_AllSamples,AFR,PAP,NEA,DEN3)

    # Function to assign putative Neanderthal blocks per population.

    def get_NEA_blocks(d3_PAP, d3_AYT, d4_PAP, d4_AYT):
        put_NEA_PAP = (d3_PAP[:,2] >= 3) & (d4_PAP[:,0] < 0)   # assign NEA if first D test has z-score >3 and 2nd D test has negative D value
        put_NEA_AYT = (d3_AYT[:,2] >= 3) & (d4_AYT[:,0] < 0)
        NEA_blocks_PAP = np.sum(put_NEA_PAP)
        NEA_blocks_AYT = np.sum(put_NEA_AYT)
        total_NEA_blocks = NEA_blocks_PAP + NEA_blocks_AYT
        shared_NEA_blocks = np.sum((put_NEA_PAP == True) & (put_NEA_AYT == True)) / total_NEA_blocks * 100   # get % of blocks that are True in both
        return shared_NEA_blocks, NEA_blocks_PAP, NEA_blocks_AYT

    shared_NEA_PAPvsAYT, putative_NEA_PAP, putative_NEA_AYT = get_NEA_blocks(D3_res_PAP, D3_res_AYT, D4_res_PAP, D4_res_AYT)

    res_arr[aRun] = D_mean_arr[0][0], D_mean_arr[1][0], shared_DEN_PAPvsAYT, shared_NEA_PAPvsAYT, mean_tract_den2, mean_tract_den1, mean_tract_nea


# In[ ]:


# function to return 2D array with dim0=(D_test_AYT, D_test_PAP, %shared_DEN, %_shared_NEA, mean_tractL_DEN2, mean_tractL_DEN1, mean_tractL_NEA) & dim1 = (mean, stdev, marginoferror)

def get_means(samplearr, interval=0.95, method='z'):
    mean_vals = np.mean(samplearr, axis=0)
    n = np.size(samplearr, axis=0)
    stdevs = np.std(samplearr, axis=0)
    if method == 't':
        test_stat=stats.t.ppf((interval+1)/2, n)
    elif method == 'z':
        test_stat=stats.norm.ppf((interval+1)/2)
    err_bounds = test_stat * stdevs / math.sqrt(n)
    alist = (mean_vals, stdevs, err_bounds)
    out_arr = np.transpose(np.stack(alist))   # stack and transpose for later plotting
    return out_arr

mean_res = get_means(res_arr)
np.savetxt('mean_res_ALT2.txt', mean_res)

# In[ ]:


# Define function to plot mean D results across 10 runs 
def plot_D_average(res_array, title=None):
    x = ['AFR,DENISOVAN(ALTAI);CEU,AYT','AFR,DENISOVAN(ALTAI);CEU,PAP']
    y = res_array[0:2,0]
    sterr = res_array[0:2,2]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.despine(ax=ax, offset=10)
    ax.errorbar(x, y, yerr=sterr)
    ax.set(ylim=(0.05, 0.30))
    ax.set_xlabel('D test comparison')
    ax.set_ylabel('Mean D test value')
    if title:
        ax.set_title(title)

# overall D test comparison
plot_D_average(mean_res, title='mean D test estimates')
plt.savefig('plot_ALT2_meanD.pdf', dpi=600, bbox_inches='tight')


# In[ ]:


# plot percentage shared blocks in population comparison

def plot_shared_blocks(res_array, title=None):
    x = ['Denisovan','Neanderthal']
    y = res_array[2:4,0]
    sterr = res_array[2:4,2]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.despine(ax=ax, offset=10)
    ax.errorbar(x, y, yerr=sterr)
    ax.set(ylim=(0, 25))
    ax.set_xlabel('Archaic population')
    ax.set_ylabel('Mean % shared putative tracts')
    if title:
        ax.set_title(title)     

plot_shared_blocks(mean_res, title='Percentage shared putative tracts between Papuan and Ayta')
plt.savefig('plot_ALT2_shared.pdf', dpi=600, bbox_inches='tight')

# plot mean tract lengths

def plot_mean_tract_lengths(res_array, title=None):
    x = ['Ayta','Papuan','Neanderthal']
    y = res_array[4:7,0]
    sterr = res_array[4:7,2]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.despine(ax=ax, offset=10)
    ax.errorbar(x, y, yerr=sterr)
#    ax.set(ylim=(0, 25))
    ax.set_xlabel('Archaic Tracts')
    ax.set_ylabel('Mean tract length')
    if title:
        ax.set_title(title)     

plot_mean_tract_lengths(mean_res, title='Mean tract lengths')
plt.savefig('plot_ALT2_meanL.pdf', dpi=600, bbox_inches='tight')

