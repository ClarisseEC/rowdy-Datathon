#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df = pd.read_csv('every-20-rows-natality.csv', low_memory=False)
df.columns = ['dt_year', 'id_state', 'id_area', 'id_cert', 'in_resident', 'id_res_status', 'id_residence', 'id_occurence', 'id_sex', 'id_attendant', 'id_f_race', 'id_m_race', 'id_c_race', 'id_c_race3', 'am_m_age', 'am_m_age36', 'am_m_age15', 'am_m_age12','am_m_age8','am_m_age7', 'am_m_age6', 'am_children_basl', 'am_children_band', 'am_children_bd', 'am_tot_b_order', 'am_tot_b_order9', 'am_live_b_order', 'am_live_b_order9', 'am_live_b_order8', 'am_live_b_order7', 'am_live_b_order6', 'am_live_b_order3', 'reserved', 'am_f_age', 'am_f_age11', 'am_birthweight', 'am_birthweight12', 'am_birthweight3', 'id_delivery_loc', 'am_plurality', 'am_plurality3', 'am_plurality2', 'dt_birth_mmdd', 'dt_last_menses', 'am_gestation', 'am_gestation10', 'am_gestation3', 'id_m_edu', 'id_m_edu14', 'id_m_edu6', 'id_f_edu', 'id_f_edu14', 'in_married', 'in_married2', 'am_prenatal', 'am_prenatal10', 'am_prenatal6', 'dt_last_lb_mm19yy', 'am_post_last_lb', 'am_post_last_lb17', 'am_post_last_lb10', 'am_post_last_lb8', 'dt_last_termin', 'in_last_termin', 'am_post_lt', 'am_post_tlp', 'am_post_tlp9', 'id_last_preg', 'id_m_birthplace', 'am_tot_prenatal', 'in_malformation', 'reserved2', 'in_report_flags', 'in_occur_flags', 'reserved3', 'id_attendant_nchs', 'am_terminationlt20', 'am_terminationgt20', 'am_1min_apgar', 'am_1min_apgar5', 'am_5min_apgar', 'am_5min_apgar5', 'id_m_descent', 'id_f_descent', 'in_gest_flag', 'id_res_region', 'id_occ_region', 'id_fips_occ', 'id_weight_rec', 'am_prenatal28', 'am_prenatal12', 'reserved4', 'id_state_occ', 'am_post_full_con', 'am_lunar_month_con', 'dt_conception', 'am_post_full_dob', 'am_lunar_month_dob', 'dt_dob', 'id', 'state', 'county', 'County FIPS', 'GHIAnnual', 'nchs_state', 'nchs_county']
pd.set_option('max_columns', None)


# In[175]:


stillborn_df = df[df['am_children_bd']==0]
stillborn_df.head()
#stillborn_df

from scipy.stats import pearsonr

#stillborn
#education of the mother
sb_m_edu = stillborn_df[stillborn_df['id_m_edu'] <= 17]

#rho = pearsonr(sb_m_edu['id_m_edu'], sb_m_edu['am_children_bd'])
# rho is the spearman rank correlation
#print(rho)

#race of the mother
#rho, p = spearmanr(stillborn_df[stillborn_df['id_m_race'] <= 0], y)
# rho is the spearman rank correlation
#print(rho)

#marital status of the mother
#rho, p = spearmanr(stillborn_df[stillborn_df['in_married'] <= 2], y)
# rho is the spearman rank correlation
#print(rho)

for i in range(1,18):
    print(i)
    print( " " )
    print(sb_m_edu[sb_m_edu['id_m_edu'] == i]['id_m_edu'].count())
#plt.show()
#plt.scatter(stillborn_df[stillborn_df['id_m_race'] <= 0], y)
#plt.show()
#plt.scatter(stillborn_df[stillborn_df['in_married'] <= 2], y)
#plt.show()
#sb_m_edu['am_children_bd']

#sb_m_edu[sb_m_edu['id_m_edu'] == 2]['id_m_edu'].count()


# In[ ]:


underweight_df = df[((df['id_sex']==1) & (df['am_birthweight']<=2547.905)) | ((df['id_sex']==2) & (df['am_birthweight']<=2526.904))]
underweight_df.head()
#underweight_df

from scipy.stats import spearmanr

#underweight
#stillborn
#education of the mother
rho, p = spearmanr(underweight_df[underweight_df['id_m_edu'] <= 17], y)
# rho is the spearman rank correlation
print(rho)
plt.scatter(underweight_df[underweight_df['id_m_edu'] <= 17], y)
plt.show()

#race of the mother
rho, p = spearmanr(underweight_df[underweight_df['id_m_race'] <= 0], y)
# rho is the spearman rank correlation
print(rho)
plt.scatter(underweight_df[underweight_df['id_m_race'] <= 0], y)
plt.show()

#marital status of the mother
rho, p = spearmanr(underweight_df[underweight_df['in_married'] <= 2], y)
# rho is the spearman rank correlation
print(rho)
plt.scatter(underweight_df[underweight_df['in_married'] <= 2], y)
plt.show()


# In[163]:


print(stillborn_df['dt_year'].value_counts())
print(underweight_df['dt_year'].value_counts())


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from decimal import Decimal

year = [1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
stillborn = [76048, 78803, 75232, 72298, 74564, 84152, 92116, 102427, 114632, 119487, 131409, 131971, 134693, 135611, 134004, 133346, 148458, 148245, 149266, 151224]
underweight = [7845, 7866, 7245, 7130, 7514, 8053, 8772, 9554, 10535, 10935, 12066, 12271, 12198, 12374, 12198, 12067, 13606, 13568, 14187, 14405]

#still born
x = year 
y = stillborn

plt.scatter(x, y)
plt.show()

#fit the model, exponential
fit = np.polyfit(x, np.log(y), 1)
c1 = math.pow(math.e, fit[1])
c2 = math.pow(math.e, fit[0])

#view the output of the model
print(fit)
print(c1)
print(c2)

#2030
finalOne = c1 * math.pow(c2, 2030)

print("The population of stillborn babies in 2030 should be: ")
print(finalOne)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

year = [1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
underweight = [7845, 7866, 7245, 7130, 7514, 8053, 8772, 9554, 10535, 10935, 12066, 12271, 12198, 12374, 12198, 12067, 13606, 13568, 14187, 14405]

#underweight babies 
x = year
y = underweight

plt.scatter(x, y)
plt.show()

#fit the model, exponential
fit = np.polyfit(x, np.log(y), 1)
c1 = math.pow(math.e, fit[1])
c2 = math.pow(math.e, fit[0])

#view the output of the model
print(fit)
print(c1)
print(c2)

#2030
finalOne = c1 * math.pow(c2, 2030)

print("The population of underweight babies in 2030 should be: ")
print(finalOne)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

year = [1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
underweight = [7845, 7866, 7245, 7130, 7514, 8053, 8772, 9554, 10535, 10935, 12066, 12271, 12198, 12374, 12198, 12067, 13606, 13568, 14187, 14405]

#underweight babies 
x = year
y = underweight

plt.scatter(x, y)

#fit the model, linear
fit = np.polyfit(x, y, 1)
m = fit[0]
c = fit[1]

#view the output of the model
x = np.array(year)
plt.plot(x, m*x+c)
print(fit)

plt.show()

#2030
finalOne = m*2030 + c

print("The population of underweight babies in 2030 should be: ")
print(finalOne)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

year = [1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
underweight = [7845, 7866, 7245, 7130, 7514, 8053, 8772, 9554, 10535, 10935, 12066, 12271, 12198, 12374, 12198, 12067, 13606, 13568, 14187, 14405]

#underweight babies 
x = year
y = underweight

plt.scatter(x, y)

#fit the model, quadratic
fit = np.polyfit(x, y, 2)
a = fit[0]
b = fit[1]
c = fit[2]

x = np.array(year)
print(x)
plt.plot(x, a*(x**2) + b*x + c)

#view the output of the model
print(fit)

plt.show()

#2030
finalOne = a*math.pow(2030, 2) + b*2030 + c

print("The population of underweight babies in 2030 should be: ")
print(finalOne)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

year = [1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
underweight = [7845, 7866, 7245, 7130, 7514, 8053, 8772, 9554, 10535, 10935, 12066, 12271, 12198, 12374, 12198, 12067, 13606, 13568, 14187, 14405]

#underweight babies 
x = year
y = underweight

plt.scatter(x, y)
plt.show()

#fit the model, cubic
fit = np.polyfit(x, y, 3)
a = fit[0]
b = fit[1]
c = fit[2]
d = fit[3]

#view the output of the model
print(fit)

#2030
finalOne = a*math.pow(2030, 3) + b*math.pow(2030, 2) + c*2030 + d

print("The population of underweight babies in 2030 should be: ")
print(finalOne)


# In[ ]:


year = [1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
stillborn = [76048, 78803, 75232, 72298, 74564, 84152, 92116, 102427, 114632, 119487, 131409, 131971, 134693, 135611, 134004, 133346, 148458, 148245, 149266, 151224]
underweight = [7845, 7866, 7245, 7130, 7514, 8053, 8772, 9554, 10535, 10935, 12066, 12271, 12198, 12374, 12198, 12067, 13606, 13568, 14187, 14405]

sum_still = 0
sum_under = 0

for i in range(0, len(stillborn)):    
   sum_still = sum_still + stillborn[i];

for i in range(0, len(underweight)):    
   sum_under = sum_under + underweight[i];   

print(sum_still)
print(sum_under)


# In[ ]:


from scipy.stats import spr

#education of the mother
rho, p = spr(df[df['id_m_edu'] <= 17], y)
# rho is the spearman rank correlation
print(rho)

#race of the mother
rho, p = spr(df[df['id_m_race'] <= 0], y)
# rho is the spearman rank correlation
print(rho)

#marital status of the mother
rho, p = spr(df[df['in_married'] <= 2], y)
# rho is the spearman rank correlation
print(rho)


# In[ ]:




