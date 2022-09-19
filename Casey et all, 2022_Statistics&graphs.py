# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:51:43 2021

@author: casey.e
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.ndimage import label, generate_binary_structure, find_objects
import matplotlib.image as mpimg
from PIL import Image
from sklearn.decomposition import PCA
from bs4 import BeautifulSoup
import requests
from io import BytesIO

#Define a function to calculate differences of means by bootstrap
def draw_bs_pairs(x, y, size=1):
    """Perform pairs bootstrap and calculate mean difference."""
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))
    # Initialize replicates: bs_diff_reps
    bs_diff_reps = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(x))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_diff_reps[i]=np.mean(bs_x-bs_y)       
    return bs_diff_reps

def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

#%%

### Fig. 1 and Supp. Fig. 1  ###

directory='https://raw.githubusercontent.com//casey-e/Casey-et-al-2022-b/main/Fig1/'
#Load dataframe for Figs. 1B,D and Supp. Fig. 1
df=pd.read_csv(directory+'Fig1B,D_data.csv', index_col=0)
#Reduce the number of AP positions by calculating the average
names=[0.94, 1.06, 1.28, 1.52]
limits=[0.7, 1.0,1.2, 1.4, 1.6]
df['AP_coordinate_avg']=pd.cut(df['AP_coordinate'], bins=limits, labels=names)
df['AP_coordinate_avg']=df['AP_coordinate_avg'].astype(float)
#Reorder regions
region_order=['CeCfd', 'CeCfv', 'CeCc','CeL', 'CeM']
df=df.set_index('Region').loc[region_order].reset_index()

### Fig. 1B

#Plot TH density across AP coordinate for every division: Fig. 1B
g=sns.lineplot(x='AP_coordinate_avg', y='Normalized_TH_intensity', data=df, hue='Division', ci=95)
plt.ylabel('TH intensity (normalized to striatum)',fontweight='bold', fontsize=12)
plt.xlabel('Anteroposterior coordinate (distance from bregma in mm)',fontweight='bold', fontsize=12)
plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
sns.despine(top=True, right=True)
plt.legend(frameon=False, prop={'size':12, 'weight':'bold'}, bbox_to_anchor=(1,1.2))
plt.ylim(0,2)
plt.title('Fig. 1B', fontsize=20, fontweight='bold',y=1.04)
plt.show()

#Statistics Fig. 1B: evaluate effect of division and AP coordinate with a linear mixed model
#Evaluate significances by comparing the Acaike information criteria of nested models
model = smf.mixedlm("Normalized_TH_intensity ~ AP_coordinate_avg * Division", df, groups="MouseId")
lm= model.fit(reml=False)
#Fit null model
null = smf.mixedlm("Normalized_TH_intensity ~ 1", df, groups="MouseId").fit(reml=False)
#Fit model with Division as factor
div= smf.mixedlm("Normalized_TH_intensity ~ Division", df, groups="MouseId").fit(reml=False)
#Fit model with Division and AP coordinate as explicatory variables, without interaction
add= smf.mixedlm("Normalized_TH_intensity ~ AP_coordinate_avg + Division", df, groups="MouseId").fit(reml=False)
#Print Acaike for each model
print('\n \n')
print('Statistics of Fig. 1B:')
print('Null model, AIC=',null.aic.round(1),';\nTH ~ Division, AIC=',div.aic.round(1), ';\nTH ~ Division + AP coordinate, AIC=',add.aic.round(1),';\nTH ~ Division x AP coordinate (with interaction), AIC=',lm.aic.round(1))


### Supp. Fig. 1
#Slice data corresponding to CeC only: df2
df2=df[df['Division']=='CeC'].groupby(['Region', 'MouseId', 'AP_coordinate_avg','Division'], as_index=False)['Normalized_TH_intensity'].mean()
df2.dropna(inplace=True)
#Make dataframe excluding data from region CeCfv: df_CeCfd
df_CeCfd=df2.drop(index=df2[df2['Region']=='CeCfv'].index).groupby(['AP_coordinate_avg'], as_index=False)['Normalized_TH_intensity'].mean()
df_CeCfd['Error']=df2.drop(index=df2[df2['Region']=='CeCfv'].index).groupby(['AP_coordinate_avg'], as_index=False)['Normalized_TH_intensity'].std()['Normalized_TH_intensity']
#Make dataframe excluding data from region df_CeCfd: df_CeCfv
df_CeCfv=df2.drop(index=df2[df2['Region']=='CeCfd'].index).groupby(['AP_coordinate_avg'], as_index=False)['Normalized_TH_intensity'].mean()
df_CeCfv['Error']=df2.drop(index=df2[df2['Region']=='CeCfd'].index).groupby(['AP_coordinate_avg'], as_index=False)['Normalized_TH_intensity'].std()['Normalized_TH_intensity']

#Plot TH intensity across AP coordinate for CeC only, separating between fronto-dorsal, fronto-ventral and caudal regions: Supp. Fig. 1
fig,ax=plt.subplots()
plt.plot(df_CeCfd['AP_coordinate_avg'], df_CeCfd['Normalized_TH_intensity'], color='steelblue')
plt.fill_between(df_CeCfd['AP_coordinate_avg'], df_CeCfd['Normalized_TH_intensity']-df_CeCfd['Error'], df_CeCfd['Normalized_TH_intensity']+df_CeCfd['Error'], alpha=0.2, color='steelblue')
plt.plot(df_CeCfv['AP_coordinate_avg'], df_CeCfv['Normalized_TH_intensity'], color='steelblue')
plt.fill_between(df_CeCfv['AP_coordinate_avg'], df_CeCfv['Normalized_TH_intensity']-df_CeCfv['Error'], df_CeCfv['Normalized_TH_intensity']+df_CeCfv['Error'], alpha=0.2, color='steelblue')
plt.scatter(df2.loc[df2['Region']=='CeCfd', 'AP_coordinate_avg'], df2.loc[df2['Region']=='CeCfd','Normalized_TH_intensity'], c='navy', label='CeCfd', marker='x')
plt.scatter(df2.loc[df2['Region']=='CeCfv', 'AP_coordinate_avg'], df2.loc[df2['Region']=='CeCfv','Normalized_TH_intensity'], c='darkcyan', label='CeCfv', marker='v')
plt.scatter(df2.loc[df2['Region']=='CeCc', 'AP_coordinate_avg'], df2.loc[df2['Region']=='CeCc','Normalized_TH_intensity'], c='slategrey', label='CeCc', marker='o')
plt.legend(frameon=False, prop={'size':12, 'weight':'bold'})
plt.ylim(0,1.2)
plt.ylabel('TH intensity (normalized to striatum)',fontweight='bold', fontsize=12)
plt.xlabel('Anteroposterior coordinate (distance from bregma in mm)',fontweight='bold', fontsize=12)
plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.title('Supp. Fig. 1', fontsize=20, fontweight='bold',y=1.04)
plt.show()


### Fig. 1C
#Process dataframe and plot scaled TH intensity across fronto-ventral axis of the frontal CeC
#Loop over the dataframes of each individual , process dataframe and normalize the TH values. Concatenate in a single dataframe: f1c_df
url = 'https://github.com/casey-e/Casey-et-al-b-2022/tree/main/Fig1/'
ext = 'csv'
df_list=[]
for file in [file[-16:] for file in listFD(url, ext) if 'Fig1C.csv' in file]:
    f1c_df=pd.read_csv(directory+file, encoding='cp1252')
    f1c_df.rename(columns={'Distance_(µm)':'Distance'}, inplace=True)
    #Normalize TH values
    f1c_df['value']=(f1c_df['Gray_Value']-f1c_df['Gray_Value'].min())*100/(f1c_df['Gray_Value'].max()-f1c_df['Gray_Value'].min())
    f1c_df.drop(columns=['Gray_Value'], inplace=True)
    f1c_df['id']=file
    df_list.append(f1c_df)
f1c_df=pd.concat(df_list, ignore_index=True)

#Plot TH intensity across dorsoventral distance: Fig. 1C
g=sns.lineplot(x='Distance', y='value', data=f1c_df, color='black', )
sns.despine(top=True, right=True)
plt.ylabel('TH intensity (bits)',fontweight='bold', fontsize=12)
plt.xlabel('Distance (um)',fontweight='bold', fontsize=12)
plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.title('Fig. 1C', fontsize=20, fontweight='bold',y=1.04)
plt.show()

#Statistics for Fig. 1C: linear mixed model
model = smf.mixedlm("value ~ Distance", f1c_df, groups="id")
lm= model.fit()
print('\n \n')
print('Statistics of Fig. 1C:')
print(lm.summary())


### Fig. 1D: bar plot and statistics for each region
#Group data by Region and Individual
df_avg=df.groupby(['Region','MouseId'], as_index=False)['Normalized_TH_intensity'].mean()
df_avg=df_avg.set_index('Region').loc[region_order].reset_index()

#Plot average TH density in each Region: Fig. 1D
g=sns.swarmplot(x='Region', y='Normalized_TH_intensity', data=df_avg, hue='MouseId', size=7)
g=sns.barplot(x='Region', y='Normalized_TH_intensity', data=df_avg, color='gray',errcolor='black',capsize=0.2, errwidth=2)
sns.despine(top=True, right=True)
plt.legend([],[], frameon=False)
plt.ylabel('TH intensity (normalized to striatum)',fontweight='bold', fontsize=15)
plt.xlabel('Region',fontweight='bold', fontsize=15)
plt.xticks(fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')
plt.title('Fig. 1D', fontsize=20, fontweight='bold',y=1.04)
plt.show()

#Statistics
print('\n \n')
print('Statistics of Fig. 1D:')
rm_aov=pg.rm_anova(data=df_avg, dv='Normalized_TH_intensity', within='Region', subject='MouseId', correction=False)
print('Repeated measures ANOVA')
print(rm_aov)
#Post-hoc Tuckey test
post_hoc=pg.pairwise_tests(data=df_avg, dv='Normalized_TH_intensity', within='Region', subject='MouseId',padjust='bonf')
print('\n')
print('Post-hoc Bonferroni')
print(post_hoc[['Contrast', 'A', 'B', 'dof','p-corr']].sort_values('p-corr').reset_index(drop=True))



#%%

### Fig. 2 ###
directory='https://raw.githubusercontent.com//casey-e/Casey-et-al-2022-b/main/Fig2/'

### Fig 2C
#Load dataframe and calculate average percentage of DAT expression in the population of TH+ neurons
df=pd.read_excel(directory+'Fig2C_data.xlsx')
df['Total TH']=df['TH+, DAT+']+df['TH+, DAT-']
df=df.groupby('MouseId', as_index=False).sum()
df['DAT+ %']=df['TH+, DAT+']/df['Total TH']
df['DAT- %']=df['TH+, DAT-']/df['Total TH']
#Plot Fig 2C
g=plt.pie([df['DAT+ %'].mean(),df['DAT- %'].mean()], labels=['TH+/DAT+','TH+/DAT-'],
          autopct='%1.1f%%',textprops ={'fontsize':25, 'fontweight':'bold'},startangle=30, labeldistance =None)
plt.legend(frameon=False, loc='upper right',bbox_to_anchor=(1.8, 1), prop={'weight':'bold', 'size':21})
plt.title('Fig. 2C', fontsize=30, fontweight='bold')
plt.show()

### Fig 2F
df=pd.read_excel(directory+'Fig2F_data.xlsx')
df=df.groupby(['MouseId', 'Area'], as_index=False).sum()
df['Dat_prop']=df['Dat']/(df['Dat']+df['DatNeg'])
df=df.sort_values('Area', ascending=False).reset_index(drop=True)
df['Dat%']=df['Dat_prop']*100

#Plot Fig. 2F
with sns.plotting_context('talk'):
    g=sns.swarmplot(x='Area', y='Dat%', data=df, hue='MouseId', size=10)
    g=sns.barplot(x='Area', y='Dat%', data=df, color='gray',errcolor='black',capsize=0.2, errwidth=2)
    sns.despine(top=True, right=True)
    plt.legend([],[], frameon=False)
    plt.ylabel('% of DAT+ retrolabeled \n DA neurons',fontweight='bold', fontsize=22)
    plt.xlabel('')
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.title('Fig. 1D', fontsize=20, fontweight='bold',y=1.04)
    plt.show()


#Statistics: p-value for mean difference in proportion of DAT+ retrolabelled DAergic neurons through bootstrap
avg_prop=df['Dat_prop'].mean()
vta_prop=np.array(df.loc[df['Area']=='VTA/SNc', 'Dat_prop'])
pag_prop=np.array(df.loc[df['Area']=='PAG/DR', 'Dat_prop'])
vta_shifted=vta_prop-np.mean(vta_prop)+avg_prop
pag_shifted=pag_prop-np.mean(pag_prop)+avg_prop


#Run bootstrap with 10000 replicates and caculate p-value: use draw_bs_pairs function
bs_diff=draw_bs_pairs(pag_shifted, vta_shifted, 10000) #Array with differences in means calculated by bootstrap
#Calculate difference of means in the actual data
real_diff=np.mean(pag_prop-vta_prop)
#Calculate p-value
p_val=np.sum(bs_diff<=real_diff)/len(bs_diff)
print('\n \n')
print('Statistics of Fig. 2F: \nBootstrap analysis to calculate p-value for mean difference in proportion of DAT+ neurons')
print('p-value (bootstrap): ', p_val)


#%%
### Fig. 4 and 5 ###

# Fig 4B
directory='https://raw.githubusercontent.com//casey-e/Casey-et-al-2022-b/main/Fig4/'
# Load dataframe and calculate average cFos per section for each region and mouse
df=pd.read_csv(directory+'Fig4B_data.csv', index_col=0)
df_means=df.groupby(['Treatment','MouseId','Region'], as_index=False)['cFos'].mean()

# Plot Fig. 4B
with sns.plotting_context("poster"):
    g=sns.barplot(x='Region', y='cFos', data=df_means, hue='Treatment', order=['CeCfd','CeCfv', 'CeCc','CeLf','CeLc', 'CeM'],hue_order=['Vehicle','Cocaine'],edgecolor='black', facecolor='gray',errcolor="black", errwidth=2, capsize=0.1)
    g=sns.swarmplot(x='Region', y='cFos', data=df_means, hue='Treatment',dodge=True,order=['CeCfd','CeCfv', 'CeCc','CeLf','CeLc', 'CeM'], size=10, hue_order=['Vehicle','Cocaine'])
    plt.ylabel('cFos+ nuclei/section', fontweight='bold')
    plt.xlabel('')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    sns.despine(top=True, right=True)
    plt.subplots_adjust(right=1.25)
    handles, labels = g.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.85,0.7), loc='lower left',frameon=False)
    g.xaxis.set_ticks_position('none')
    plt.title('Fig. 4B', fontsize=30, fontweight='bold',y=1)
    plt.show()

# #The following lines are for making and saving the dataframe used for statistics in R

# df_sum=df.groupby(['Treatment','MouseId','Region'], as_index=False)['cFos'].sum()
# df_offset=df.reset_index().groupby(['Treatment','MouseId','Region'], as_index=False)['index'].nunique().rename(columns={'index':'Sections'})
# df_sum=df_sum.merge(df_offset, how='inner', on=['Treatment','MouseId','Region'],suffixes=('', '') ,validate='one_to_one')
# df_sum.to_csv('Fig4B_Statistics.csv')

### Fig. 4D
# Load dataframe, convert to long format and calculate average cFos per section for each region and mouse
df=pd.read_csv(directory+'Fig4D_data.csv', index_col=0)
df=df.melt(id_vars=['MouseId', 'Treatment', 'AP_coordinate_neg'], var_name='Region', value_name='cFos')
df.dropna(subset=['cFos'], inplace=True)
df_means=df.groupby(['Treatment','MouseId','Region'], as_index=False)['cFos'].mean()

# Plot Fig. 4D
with sns.plotting_context("poster"):
    g=sns.barplot(x='Region', y='cFos', data=df_means, hue='Treatment', order=['CeCfd','CeCfv', 'CeCc','CeLf','CeLc', 'CeM'],hue_order=['Vehicle','Quinpirole','Haloperidol','SKF'],edgecolor='black', facecolor='gray',errcolor="black", errwidth=2, capsize=0.1)
    g=sns.swarmplot(x='Region', y='cFos', data=df_means, hue='Treatment',dodge=True,order=['CeCfd','CeCfv', 'CeCc','CeLf','CeLc', 'CeM'], size=10, hue_order=['Vehicle','Quinpirole','Haloperidol','SKF'])
    plt.ylabel('cFos+ nuclei/section', fontweight='bold')
    plt.xlabel('')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    sns.despine(top=True, right=True)
    plt.subplots_adjust(right=2)
    handles, labels = g.get_legend_handles_labels()
    plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.9,0.99), loc='upper left',frameon=False)
    g.xaxis.set_ticks_position('none')
    plt.title('Fig. 4D', fontsize=30, fontweight='bold',y=1)
    plt.show()

# #The following lines are for making and saving the dataframe used for statistics in R

# df_sum=df.groupby(['Treatment','MouseId','Region'], as_index=False)['cFos'].sum()
# df_offset=df.reset_index().groupby(['Treatment','MouseId','Region'], as_index=False)['index'].nunique().rename(columns={'index':'Sections'})
# df_sum=df_sum.merge(df_offset, how='inner', on=['Treatment','MouseId','Region'],suffixes=('', '') ,validate='one_to_one')
# df_sum.to_csv('Fig4D_Statistics.csv')


### Fig. 5
directory='https://raw.githubusercontent.com//casey-e/Casey-et-al-2022-b/main/Fig5/'

treatment_list=['Cocaine','SKF','Quinpirole','Haloperidol'] #List of treatments
AP_list=[0.94,1.06,1.34,1.58] #List of antero-posterior coordinates

#The following dictionaries are for slicing the areas corresponding to the samples to quantify cFos. Each dctionary corresponds to an antero-posterior coordinate 
crops_dict_094={'CeCfd1':[130,60], 'CeCfd2':[350, 44],'CeCfv1':[600, 100],'CeCfv2':[550, 190], 'CeLf1':[165, 230], 'CeLf2':[250,300], 'CeM1':[400, 300], 'CeM2':[500, 400]}
crops_dict_106={'CeCfd3':[150,55], 'CeCfd4':[300, 40],'CeCfv3':[450, 150],'CeCfv4':[570,130], 'CeLf3':[90, 240], 'CeLf4':[190,240], 'CeM3':[400, 300], 'CeM4':[450, 400]}
crops_dict_134={'CeCc1':[200,70], 'CeCc2':[350, 70], 'CeLc1':[180, 210], 'CeLc2':[300,180], 'CeM5':[300, 350], 'CeM6':[500, 300]}
crops_dict_158={'CeCc3':[200,70], 'CeCc4':[350, 70], 'CeLc3':[180, 210], 'CeLc4':[300,180], 'CeM7':[300, 350], 'CeM8':[500, 300]}
#The following dict links each prior dictionary with and antero-posterior coordinate
AP_crops_dict={'0.94':crops_dict_094,'1.06':crops_dict_106, '1.34':crops_dict_134, '1.58':crops_dict_158}


#The following for loop iterates over each image, and draws squares showing the sampling areas in images of cFos and TH 
# immunostaining so the studied samples can be vissually verified.
# In addition, for each image of cFos immunostaining, it slices the sampling areas, count cFos positive nuclei using
# a treshold of value (intensity of fluorescence) and size, and store the counts in a list of dataframes, where each
# dataframe corresponds to an image.
url = 'https://github.com/casey-e/Casey-et-al-b-2022/tree/main/Fig5/'
ext = 'tif'
df_list=[] #List to store dataframes with the number of cFos per sampling area, for each image of cFos immunostaining
# for file in glob('*.tif'):
for file in [file[43:] for file in listFD(url, ext)]:
    #Define the variable treatment depending on the name of the image
    for i,j in zip(['C','S','Q','H'],treatment_list):
        if i in file:
            treatment=j
    #Define MouseId variable
    mouseId=file[11:14].replace('_','') 
    #Define AP_position variable
    for ap in AP_list:
        if str(ap) in file:
            AP_position=ap
    #Open the image
    response = requests.get(directory+file)
    im=Image.open(BytesIO(response.content))
    # Load the image as numpy array
    img = np.array(im)
    # Keep only the  channel of interest (as images were taken using a fluorescence microscope)
    img2=(img[:,:,1])
    #Show image with its anteroposterior coordinate, treatment and MouseId
    plt.imshow(img2)
    plt.title(str(AP_position)+'_'+treatment+'_'+mouseId,fontsize=16, fontweight='bold')
    plt.show()
    #Transform the numpy array of the image to a pandas dataframe
    img_df=pd.DataFrame(img2).copy()
    
    #Choose the appropiate crops_dict (to slice the samplimg areas) based on the  name of the image
    for key,value in AP_crops_dict.items():
        if key in file:
            crops_dict=value
    
    # Draw squares to show the sampling areas
    for key, value in crops_dict.items():
        heigh=list(range(value[0]-40,value[0]+40)) #Define heigh
        width=list(range(value[1]-40,value[1]+40)) #Define width
        # Draw the squares by replacing the values in the pandas dataframe by a very high value, using heigh and width
        img_df.loc[heigh,width[:3]]=600
        img_df.loc[heigh,width[-3:]]=600
        img_df.loc[heigh[:3],width]=600
        img_df.loc[heigh[-3:],width]=600
    img5=np.array(img_df) #Convert the modified datarame to numpy array: img5
    #Show image with sampling areas
    plt.imshow(img5, vmin=img2.min(), vmax=img2.max())
    plt.title('AP: -'+str(AP_position)+' mm', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.xlabel('')
    plt.yticks([])
    plt.xticks([])
    plt.show()

    #Slice sampling areas and quantify cFos expressing nuclei
    if 'fos' in file: #Only quantify images corresponding to cFos, discard images of TH immunofluorescence
        img2b=img2.copy()
        #Set a treshold: the values in the highest 0.5% are transformed to 1, the rest is transformed to 0
        img3=np.where(img2b > np.percentile(img2b,99.5),1,0)
        #Show filtered image
        plt.imshow(img3)
        plt.title('AP: -'+str(AP_position)+' mm', fontsize=16, fontweight='bold')
        plt.ylabel('')
        plt.xlabel('')
        plt.yticks([])
        plt.xticks([])
        plt.show()
        #Loop over sampling areas, count cFos nuclei and store it in counts_dict
        counts_dict={}
        for key, value in crops_dict.items():
            heigh=list(range(value[0]-40,value[0]+40))#Set heigh
            width=list(range(value[1]-40,value[1]+40))#Set width
            #Slice sampling area using label function and heigh and width variables
            labels,names=label(img3[np.ix_(heigh,width)])
            #Count cFos nuclei in the sampling area
            slices=find_objects(labels)#find ojects:slices
            count=0#Initiate counter: count
            #Iterate along slices and add 1 to count only if the object size is higher than a teshold of 30 (empirical value)
            for slic in slices:
                if labels[slic].size >=30:
                    count+=1
            #Show the tresholded sampling area with the number of cFos counted
            plt.imshow(labels)
            plt.title(treatment+'_'+mouseId+'_'+key+'_counts='+str(count))
            plt.show()
            
            counts_dict.update({key:count})#Add the counted value to counts_dict
        #Make a pandas dataframe with counts_dict and add MouseId, Treatment and AP_position columns
        df=pd.DataFrame(counts_dict, index=[0])
        df['MouseId']=mouseId
        df['Treatment']=treatment
        df['AP_position']=AP_position
        #Append the dataframe of the current image to df_list
        df_list.append(df)
#Concatenate the dataframes of fd_list: df
df=pd.concat(df_list, ignore_index=True)
#Tranform df to long format and add "Region" and "Division" columns: df2
df2=df.melt(id_vars=['MouseId', 'Treatment', 'AP_position'], var_name='Sample', value_name='cFos')
df2.dropna(inplace=True)
df2['Region']=df2['Sample'].str[:-1]
df2['Division']=df2['Region'].str[0:3]

### Fig. 4E 
#Prepare dataframe: modify d2f to be able to plot CeCfd-CeCc and CeCfv-CeCc separately
#Make dataframe with CeCfv and CeCc only: CeCfv_CeCc
CeCfv_CeCc=df2[(df2['Division']=='CeC')&(df2['Region']!='CeCfd')]
CeCfv_CeCc['Division']='CeCfv-CeCc'
#Make dataframe with CeCfd and CeCc only: CeCfd_CeCc
CeCfd_CeCc=df2[(df2['Division']=='CeC')&(df2['Region']!='CeCfv')]
CeCfd_CeCc['Division']='CeCfd-CeCc'
#Concatenate df2 (excluding CeC), CeCfv_CeCc and CeCfd_CeCc to make df2b
df2b=pd.concat([df2[df2['Division']!='CeC'], CeCfv_CeCc,CeCfd_CeCc], ignore_index=True)
#Calculate average acros samples of the same region, for each antero-posterior coordinate, mouse and treatment
df2b=df2b.groupby(['Treatment','MouseId','AP_position','Division','Region'], as_index=False)['cFos'].mean()

#Plot Fig. 4E
with sns.plotting_context("poster"):
    g=sns.relplot(x='AP_position', y='cFos',data=df2b, kind='line', hue='Treatment', style='Treatment',col='Division', markers=True, col_order=['CeCfd-CeCc','CeCfv-CeCc','CeL','CeM'],hue_order=['Cocaine','Quinpirole','Haloperidol','SKF'])
    g.map(plt.axvline,x=(0.94+(1.58-0.94)/2),ls="--",c="black")
    g.map(plt.annotate,text='Frontal', xy=(0.96,5.5), color='black')
    g.map(plt.annotate,text='Caudal', xy=(1.4,5.7), color='black')
    g.set_titles('{col_name}', size=25, fontweight='bold')
    plt.suptitle('Fig. 4E', size=35, fontweight='bold', y=1.1)
    g.set_axis_labels('AP coordinate\n(-1x from Bregma)', 'Number of cFos', fontweight='bold')
    plt.show()


### Principal component analysis and Fig. 5B-C

#Calculate mean value of each sample across individuals, for each treatment:df3
df3=df2.groupby(['Treatment', 'Region','AP_position', 'Sample'], as_index=False)['cFos'].mean()
#Prepare dataframe for principal component analysis (PCA)
df3=df3.drop(columns=['Region']).pivot_table(index=['AP_position','Sample'], columns='Treatment',values='cFos').dropna().reset_index()
df3['Region']=df3['Sample'].str[:-1]
df3['Division']=df3['Region'].str[0:3]
#Make numpy array with values of cFos corresponding to each treatment in df3: data
data=df3[treatment_list].values
#Run PCA on data array and store values of principal components in the array pc
pca=PCA(n_components=3)
pc=pca.fit(data).transform(data)
#Add values of principal components to df3: "PC1","PC2","PC3"
df3['PC1']=pc[:,0]
df3['PC2']=pc[:,1]
df3['PC3']=pc[:,2]

print('\n\n Results Principal Components Analysis, Fig. 5B-C')

print('\n\nExplained variance by each principal component:\nPC1: ',round(pca.explained_variance_ratio_[0]*100, 1),
      '\nPC2: ',round(pca.explained_variance_ratio_[1]*100,1),'\nPC3: ',round(pca.explained_variance_ratio_[2]*100,1))

components_df=pd.DataFrame(pca.components_, columns=treatment_list, index=['PC1','PC2','PC3']).round(2)
print('\nCoeficients (contribution of each original variable to each principal component):')
print(components_df)



## Plot Fig. 5B

# Left
with sns.plotting_context("poster"):
    g=sns.relplot(x='PC1', y='PC2', data=df3, kind='scatter', hue='Region', style='Division')
    plt.suptitle('Fig.5B, left:\nPC1 v. PC2', fontweight='bold',y=1.1)
    plt.show()
# Right
with sns.plotting_context("poster"):
    g=sns.relplot(x='PC3', y='PC2', data=df3, kind='scatter', hue='Region', style='Division')
    plt.suptitle('Fig.5B, right:\nPC3 vs. PC2', fontweight='bold',y=1.1)
    plt.show()


# Plot Fig 5C: 3D scatterplot
region_list=df3['Region'].unique()
color_list=sns.color_palette(as_cmap=True)[0:6]
division_list=df3['Division'].unique()
markers_list=['o','x','^']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for division, marker in zip(division_list,markers_list):
    for region, color in zip(region_list,color_list):
        val=df3.loc[(df3['Division']==division)&(df3['Region']==region), ['PC1', 'PC2', 'PC3']].values
        if len(val)>0:
            ax.scatter(val[:,0], val[:,1], val[:,2], marker=marker, color=color, label=region)
plt.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.view_init(-140, 60)
plt.title('Fig. 5C', fontweight='bold')
plt.show()