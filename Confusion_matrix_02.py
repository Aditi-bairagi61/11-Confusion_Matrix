# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:57:15 2024

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
#target value (true value)
true_value=50

#simulate data
#1. accurate and precise (close to true value and tightly grouped)

'''loc=true_value (true value=50): the values will be 
centered around the true value(50).
scale=1 : the standard deviation (spread ) is small
meaning the values will be tightly grouped around the true values
this implies high precision,
the measurements will vary only a little from the true value
so they'll be both accurate (close to 50) and
precise (close to each other)
'''
accurate_precise=np.random.normal(loc=true_value,scale=1,size=10)

#2.accurate but not precise (close to true value but spread out)
accurate_not_precise=np.random.normal(loc=true_value,scale=10,size=10,)
''' the two lines of code you have highlighted  may look similar,
but they differ in one important aspect, the value of scale
which controls the spread of the generated values around
the true value(loc)
'''

#3. precise but not accurate (far from true value but tightly grouped)
precise_not_accurate=np.random.normal(loc=70, scale=1, size=10)

#4. neither accurate nor precise (far from true value and spread out)
not_accurate_not_precise=np.random.normal(loc=70, scale=10, size=10)

#plotting the results
plt.figure(figsize=(10,6))

#plot 1: accurate and precise
plt.scatter(accurate_precise,[1]*10,color='green',label="Accurate and Precise")

#plot 2: accurate but not precise
plt.scatter(accurate_not_precise,[2]*10,color='blue', label="Accurate but not precise")

#plot 3: precise but not accurate
plt.scatter(precise_not_accurate,[3]*10,color='orange', label="Precise but not Accurate")

#plot 4: neither accurate not precise
plt.scatter(not_accurate_not_precise,[4]*10,color='red', label="neither Accurate nor precise")

#adding targer line
plt.axvline(true_value,color='black', linestyle = '--',label='true value')

#labels and legend
plt.yticks([1,2,3,4],['Accurate and precise', 'Accurate but not precise', 'precise but not accurate', 'neither accurate nor precise'])

plt.xlabel('Measurement Value')
plt.legend()
plt.title('Accurate and precision demonstration ')
plt.show()
