#Petter Eriksson, 2024-10-8, peer22@student.bth.se
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'task_1\GA_TSP_Results.csv')  # Using raw string to avoid escape sequence issues

# Calculate median distance for each population size
median_pop_size = data.groupby('Population Size')['Best Distance'].median().reset_index()

# Calculate median distance for each mutation rate
median_mutation_rate = data.groupby('Mutation Rate')['Best Distance'].median().reset_index()

# Line Plot, Mutation Rate vs. Best Distance (Grouped by Population Size)
plt.figure(figsize=(10, 6))
for pop_size in data['Population Size'].unique():
    subset = data[data['Population Size'] == pop_size]
    plt.plot(subset['Mutation Rate'], subset['Best Distance'], marker='o', label=f'Population Size {pop_size}')
plt.xlabel('Mutation Rate')
plt.ylabel('Best Distance')
plt.title('Mutation Rate vs. Best Distance (Grouped by Population Size)')
plt.legend()
plt.grid(True)
plt.show()

# Bar Plot, Population Size and Mutation Rate vs. Best Distance
plt.figure(figsize=(12, 8))
sns.barplot(x='Population Size', y='Best Distance', hue='Mutation Rate', data=data)
plt.xlabel('Population Size')
plt.ylabel('Best Distance')
plt.title('Population Size and Mutation Rate vs. Best Distance')
plt.legend(title='Mutation Rate')
plt.show()

# Scatter Plot, Population Size and Mutation Rate vs. Best Distance
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['Mutation Rate'], data['Best Distance'], c=data['Population Size'], cmap='viridis', s=100)
plt.colorbar(scatter, label='Population Size')
plt.xlabel('Mutation Rate')
plt.ylabel('Best Distance')
plt.title('Scatter Plot: Mutation Rate and Best Distance vs. Population Size')
plt.grid(True)
plt.show()

print("\nMedian Best Distances for each Population Size:")
print(median_pop_size)

print("\nMedian Best Distances for each Mutation Rate:")
print(median_mutation_rate)