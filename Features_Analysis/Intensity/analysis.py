import pandas as pd


def analyze_species_darkness():
    # Load data
    darkness_df = pd.read_csv("Darkness_Analysis_Results/wingtip_darkness_analysis.csv")

    # Calculate species averages
    species_avg = (darkness_df.groupby('species')['percentage_darker']
    .agg(['mean', 'std', 'count'])
    .reset_index()
    .rename(columns={
        'mean': 'avg_percentage_darker',
        'std': 'std_deviation',
        'count': 'sample_count'
    }))

    # Save results
    species_avg.to_csv("Darkness_Analysis_Results/species_comparison.csv", index=False)
    print("Species comparison saved to species_comparison.csv")

    return darkness_df, species_avg


if __name__ == "__main__":
    full_results, species_results = analyze_species_darkness()
    print("\nSpecies Comparison Results:")
    print(species_results)
