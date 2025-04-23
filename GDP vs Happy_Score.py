
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
data = pd.read_csv('happyscore_income.csv')


# In[2]:


happy = data['happyScore']
gdp = data['GDP']
gdp_mean = np.mean(gdp)
happy_mean = np.mean(happy)
gdp_stddev = np.std(gdp)
happy_stddev = np.std(happy)


ellipse = patches.Ellipse([gdp_mean, happy_mean], gdp_stddev*2, happy_stddev*2, alpha = 0.2)

# Find the country with lowest happyScore where GDP > 1.0
lowest_happy_high_gdp = data[data['GDP'] > 1.0].nsmallest(1, 'happyScore')
country_name = lowest_happy_high_gdp['country'].values[0]
country_gdp = lowest_happy_high_gdp['GDP'].values[0]
country_happy = lowest_happy_high_gdp['happyScore'].values[0]


# In[3]:


fig,graph = plt.subplots()
plt.scatter(gdp , happy)
plt.scatter(gdp_mean, happy_mean)
graph.add_patch(ellipse)
plt.title('Happiness Score vs GDP', pad=20)
plt.xlabel('GDP')
plt.ylabel('Happy Score')
plt.text(country_gdp, country_happy, country_name, 
         ha='left', va='bottom')

