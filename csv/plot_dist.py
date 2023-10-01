import matplotlib.pyplot as plt
import seaborn as sns


# Your Data
list_1 = [1,2,3,4,5,1,2,3,4,5,6,3,4,5,1,3,4,5,4,5,6,8,9,12,3,3,3,4,3,4,5,6,5,6,7,8,9,5,3,2,4,5,2,3,4,11,13,4,5,3,5,6,7,11,13,3,4,5,4,5]
list_2 = [4,5,6,7,8,9,4,5,6,7,8,9,5,6,7,8,9,6,7,8,9,12,15,16,11,12,7,8,9,7,8,9,5,6,7,8,9,7,8,9,8,9,11,10,12,16,7,8,9,10,10,8,9,8,9,10,10,10,15,16,19]

# Creating a displot
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)

sns.distplot(list_1, kde=True, ax = ax, hist=False, bins = 10)
sns.distplot(list_2, kde=True, ax = ax, hist=False, bins = 10)

plt.show()