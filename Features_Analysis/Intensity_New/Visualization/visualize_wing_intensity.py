import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Get the project root directory - fix path handling
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent.parent

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Custom color palette
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

def load_data():
    """Load wing intensity data from CSV file."""
    try:
        # Try multiple possible paths
        possible_paths = [
            PROJECT_ROOT / "Wing_Intensity_Results_New" / "wing_intensity_analysis.csv",
            PROJECT_ROOT / "Wing_Intensity_Results_New" / "wing_intensity_averages.csv",
            PROJECT_ROOT / "Features_Analysis" / "Intensity_New" / "wing_intensity_analysis.csv",
            PROJECT_ROOT / "Features_Analysis" / "Intensity_New" / "wing_intensity_averages.csv"
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
                
        if data_path is None:
            # Print all available files in the directories to help debug
            print("Available files in Wing_Intensity_Results_New:")
            wing_dir = PROJECT_ROOT / "Wing_Intensity_Results_New"
            if wing_dir.exists():
                for file in wing_dir.glob("*"):
                    print(f"  - {file.name}")
            
            print("\nAvailable files in Features_Analysis/Intensity_New:")
            features_dir = PROJECT_ROOT / "Features_Analysis" / "Intensity_New"
            if features_dir.exists():
                for file in features_dir.glob("*"):
                    print(f"  - {file.name}")
                    
            raise FileNotFoundError(f"Data file not found. Tried paths: {[str(p) for p in possible_paths]}")
        
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        print(f"Successfully loaded data with {len(data)} rows")
        print("\nColumns in dataset:", data.columns.tolist())
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def create_output_dir():
    """Create output directory for plots."""
    try:
        output_dir = PROJECT_ROOT / "Wing_Intensity_Results_New" / "Visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        sys.exit(1)

def plot_intensity_distribution(data, output_dir):
    """Plot distribution of wing intensities for each species."""
    try:
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Violin plot
        ax1 = plt.subplot(gs[0, 0])
        sns.violinplot(data=data, x='species', y='mean_intensity', 
                      inner='box', cut=0, palette=colors)
        sns.stripplot(data=data, x='species', y='mean_intensity',
                     size=4, alpha=0.3, jitter=True)
        ax1.set_title('Distribution of Wing Intensities')
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Mean Intensity')
        
        # Box plot
        ax2 = plt.subplot(gs[0, 1])
        sns.boxplot(data=data, x='species', y='mean_intensity', palette=colors)
        sns.stripplot(data=data, x='species', y='mean_intensity',
                     size=4, alpha=0.3, jitter=True)
        ax2.set_title('Wing Intensity Comparison')
        ax2.set_xlabel('Species')
        ax2.set_ylabel('Mean Intensity')
        
        # Histogram
        ax3 = plt.subplot(gs[1, :])
        for species, color in zip(data['species'].unique(), colors):
            species_data = data[data['species'] == species]
            sns.histplot(data=species_data, x='mean_intensity', 
                        bins=30, alpha=0.5, label=species, color=color)
        ax3.set_title('Histogram of Wing Intensities')
        ax3.set_xlabel('Mean Intensity')
        ax3.set_ylabel('Count')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intensity_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Created intensity distribution plot")
    except Exception as e:
        print(f"Error creating intensity distribution plot: {str(e)}")

def plot_intensity_statistics(data, output_dir):
    """Create statistical summary plots."""
    try:
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Mean and std
        ax1 = plt.subplot(gs[0, 0])
        stats_data = data.groupby('species').agg({
            'mean_intensity': ['mean', 'std'],
            'median_intensity': 'mean',
            'skewness': 'mean',
            'kurtosis': 'mean'
        }).round(2)
        
        stats_data.columns = ['Mean', 'Std', 'Median', 'Skewness', 'Kurtosis']
        sns.heatmap(stats_data, annot=True, cmap=custom_cmap, ax=ax1)
        ax1.set_title('Statistical Summary')
        
        # Intensity ranges
        ax2 = plt.subplot(gs[0, 1])
        range_data = data.groupby('species').agg({
            'min_intensity': 'min',
            'max_intensity': 'max'
        })
        range_data['range'] = range_data['max_intensity'] - range_data['min_intensity']
        sns.barplot(data=range_data.reset_index(), x='species', y='range', 
                   palette=colors, ax=ax2)
        ax2.set_title('Intensity Range by Species')
        ax2.set_xlabel('Species')
        ax2.set_ylabel('Intensity Range')
        
        # Percentile distribution
        ax3 = plt.subplot(gs[1, :])
        percentile_cols = [col for col in data.columns if col.startswith('pct_')]
        percentile_data = data.groupby('species')[percentile_cols].mean()
        sns.heatmap(percentile_data, annot=True, cmap=custom_cmap, ax=ax3)
        ax3.set_title('Percentile Distribution of Intensities')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intensity_statistics.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Created intensity statistics plot")
    except Exception as e:
        print(f"Error creating intensity statistics plot: {str(e)}")

def plot_intensity_correlation(data, output_dir):
    """Create correlation plots."""
    try:
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Correlation matrix
        ax1 = plt.subplot(gs[0, :])
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation = data[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax1)
        ax1.set_title('Correlation Matrix')
        
        # Scatter plots
        ax2 = plt.subplot(gs[1, 0])
        sns.scatterplot(data=data, x='mean_intensity', y='std_intensity', 
                       hue='species', palette=colors, ax=ax2)
        ax2.set_title('Mean vs Standard Deviation')
        
        ax3 = plt.subplot(gs[1, 1])
        sns.scatterplot(data=data, x='median_intensity', y='mean_intensity', 
                       hue='species', palette=colors, ax=ax3)
        ax3.set_title('Median vs Mean Intensity')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intensity_correlation.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Created intensity correlation plot")
    except Exception as e:
        print(f"Error creating intensity correlation plot: {str(e)}")

def perform_statistical_tests(data):
    """Perform statistical tests to compare species."""
    try:
        species_groups = [group for _, group in data.groupby('species')]
        
        if len(species_groups) < 2:
            raise ValueError("Need at least 2 species for statistical comparison")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            species_groups[0]['mean_intensity'],
            species_groups[1]['mean_intensity']
        )
        
        # Perform Mann-Whitney U test
        u_stat, u_p_value = stats.mannwhitneyu(
            species_groups[0]['mean_intensity'],
            species_groups[1]['mean_intensity'],
            alternative='two-sided'
        )
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(
            species_groups[0]['mean_intensity'],
            species_groups[1]['mean_intensity']
        )
        
        return {
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_p_value},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p_value}
        }
    except Exception as e:
        print(f"Error performing statistical tests: {str(e)}")
        return None

def plot_statistical_summary(data, stats_results, output_dir):
    """Create summary plot with statistical test results."""
    try:
        if stats_results is None:
            print("Skipping statistical summary plot due to missing test results")
            return
            
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Box plot with statistical annotations
        ax1 = plt.subplot(gs[0, :])
        sns.boxplot(data=data, x='species', y='mean_intensity', palette=colors)
        sns.stripplot(data=data, x='species', y='mean_intensity',
                     size=4, alpha=0.3, jitter=True)
        
        y_max = data['mean_intensity'].max()
        stats_text = (
            f"T-test p-value: {stats_results['t_test']['p_value']:.4f}\n"
            f"Mann-Whitney U p-value: {stats_results['mann_whitney']['p_value']:.4f}\n"
            f"Kolmogorov-Smirnov p-value: {stats_results['kolmogorov_smirnov']['p_value']:.4f}"
        )
        plt.text(0.5, y_max * 1.1, stats_text, ha='center', va='bottom')
        
        ax1.set_title('Wing Intensity Comparison with Statistical Tests')
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Mean Intensity')
        
        # QQ plot
        ax2 = plt.subplot(gs[1, 0])
        stats.probplot(data['mean_intensity'], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Mean Intensities')
        
        # Distribution plot
        ax3 = plt.subplot(gs[1, 1])
        for species, color in zip(data['species'].unique(), colors):
            species_data = data[data['species'] == species]
            sns.kdeplot(data=species_data['mean_intensity'], 
                       label=species, color=color, ax=ax3)
        ax3.set_title('Kernel Density Estimation')
        ax3.set_xlabel('Mean Intensity')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_summary.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Created statistical summary plot")
    except Exception as e:
        print(f"Error creating statistical summary plot: {str(e)}")

def main():
    print("Starting wing intensity visualization...")
    
    # Load data
    data = load_data()
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Generate plots
    plot_intensity_distribution(data, output_dir)
    plot_intensity_statistics(data, output_dir)
    plot_intensity_correlation(data, output_dir)
    
    # Perform statistical tests
    stats_results = perform_statistical_tests(data)
    
    # Create statistical summary plot
    plot_statistical_summary(data, stats_results, output_dir)
    
    # Print statistical results
    if stats_results:
        print("\nStatistical Test Results:")
        print(f"T-test p-value: {stats_results['t_test']['p_value']:.4f}")
        print(f"Mann-Whitney U p-value: {stats_results['mann_whitney']['p_value']:.4f}")
        print(f"Kolmogorov-Smirnov p-value: {stats_results['kolmogorov_smirnov']['p_value']:.4f}")
    
    # Print summary statistics
    print("\nSummary Statistics by Species:")
    print(data.groupby('species')['mean_intensity'].describe())
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 