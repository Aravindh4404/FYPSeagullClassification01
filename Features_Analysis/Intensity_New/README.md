# Intensity Analysis for Seagull Classification

This directory contains a new implementation of the intensity analysis for seagull wing and wingtip classification. The implementation is designed to be consistent, reliable, and avoid double normalization issues.

## Files

- `normalization_utils.py`: Contains utility functions for image normalization and region extraction
- `wing_intensity_analysis.py`: Analyzes wing intensity for each image
- `wingtip_darkness_analysis.py`: Analyzes wingtip darkness compared to wing mean intensity
- `run_intensity_analysis.py`: Runs the complete analysis pipeline

## Key Features

1. **Consistent Normalization**: All images are normalized once using the same method (`minmax_normalize`)
2. **Global Normalization**: The entire image is normalized before extracting regions
3. **Comprehensive Metrics**: Calculates various intensity metrics including:
   - Basic statistics (mean, std, min, max, median)
   - Distribution metrics (skewness, kurtosis)
   - Intensity range counts
   - Darker pixel percentages
   - Threshold-based metrics (pixels darker than specific thresholds)

## Usage

To run the complete analysis:

```bash
python run_intensity_analysis.py
```

This will:
1. Run wing intensity analysis
2. Run wingtip darkness analysis
3. Save results to:
   - `Wing_Intensity_Results_New/`
   - `Wingtip_Darkness_Results_New/`

## Output Files

### Wing Intensity Analysis
- `wing_intensity_analysis.csv`: Detailed wing intensity metrics for each image
- `wing_intensity_analysis.pkl`: Same data in pickle format
- `wing_intensity_averages.csv`: Species averages for wing intensity metrics

### Wingtip Darkness Analysis
- `wingtip_darkness_analysis.csv`: Detailed wingtip darkness metrics for each image
- `wingtip_darkness_analysis.pkl`: Same data in pickle format
- `wingtip_darkness_averages.csv`: Species averages for wingtip darkness metrics

## Metrics Explained

### Wing Intensity Metrics
- `mean_intensity`: Average intensity of wing pixels
- `std_intensity`: Standard deviation of wing pixel intensities
- `intensity_X_Y`: Count of pixels in intensity range [X, Y)
- `pct_X_Y`: Percentage of pixels in intensity range [X, Y)

### Wingtip Darkness Metrics
- `percentage_darker`: Percentage of wingtip pixels darker than wing mean
- `mean_darker_wingtip_intensity`: Mean intensity of darker wingtip pixels
- `diff_gt_X`: Count of wingtip pixels darker than wing mean by X
- `pct_diff_gt_X`: Percentage of wingtip pixels darker than wing mean by X
- `dark_lt_X`: Count of wingtip pixels with intensity less than X
- `pct_dark_lt_X`: Percentage of wingtip pixels with intensity less than X 