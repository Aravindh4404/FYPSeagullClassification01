import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('Darkness_Analysis_Results/wingtip_darkness_analysis.csv')

# Calculate average percentage darker for each species
species_avg = df.groupby('species')['percentage_darker'].mean().reset_index()
species_avg = species_avg.sort_values('percentage_darker', ascending=False)

# Print results
print("Average percentage darker by species:")
for _, row in species_avg.iterrows():
    print(f"{row['species']}: {row['percentage_darker']:.2f}%")

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(species_avg['species'], species_avg['percentage_darker'])
plt.title('Average Percentage of Darker Wingtip Pixels by Species')
plt.xlabel('Species')
plt.ylabel('Average Percentage Darker')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('species_darkness_comparison.png')
plt.close()

# Calculate overall statistics
print("\nOverall statistics:")
print(f"Total images analyzed: {len(df)}")
print(f"Overall average percentage darker: {df['percentage_darker'].mean():.2f}%")
print(f"Minimum percentage darker: {df['percentage_darker'].min():.2f}%")
print(f"Maximum percentage darker: {df['percentage_darker'].max():.2f}%")
