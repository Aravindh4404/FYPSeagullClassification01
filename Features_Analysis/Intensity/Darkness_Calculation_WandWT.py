import os
import pandas as pd

def analyze_wingtip_darkness():
    # Load the wing and wingtip intensity data from CSV files
    wing_csv = pd.read_csv('Features_Analysis/Intensity/Wing_Greyscale_Intensity_Results/wing_intensity_analysis.csv')
    wingtip_csv = pd.read_csv('Features_Analysis/Intensity/Wingtip_Greyscale_Intensity_Results/wingtip_intensity_analysis.csv')
    
    # Calculate the mean intensity of the wing region
    wing_mean_intensity = wing_csv['mean_intensity'].mean()
    print(f"Mean intensity of wing region: {wing_mean_intensity:.2f}")
    
    # Identify wingtip pixels that are darker than the mean intensity of the wing region
    wingtip_darker_pixels = wingtip_csv[wingtip_csv['mean_intensity'] < wing_mean_intensity]
    
    # Calculate the percentage of wingtip pixels that are darker than the mean wing intensity
    wingtip_total_pixels = wingtip_csv['pixel_count'].sum()
    wingtip_darker_pixels_count = wingtip_darker_pixels['pixel_count'].sum()
    percentage_darker_pixels = (wingtip_darker_pixels_count / wingtip_total_pixels) * 100
    
    print(f"Total wingtip pixels: {wingtip_total_pixels}")
    print(f"Darker wingtip pixels: {wingtip_darker_pixels_count}")
    print(f"Percentage of darker pixels: {percentage_darker_pixels:.2f}%")
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a directory to save the results
    output_dir = os.path.join(current_dir, 'Darkness_Analysis_Results')
    os.makedirs(output_dir, exist_ok=True)

    # Save the list of darker wingtip pixels to a CSV file
    output_csv_path = os.path.join(output_dir, 'wingtip_darker_pixels_analysis.csv')
    wingtip_darker_pixels.to_csv(output_csv_path, index=False)
    
    # Save the summary statistics to a text file
    output_txt_path = os.path.join(output_dir, 'darkness_analysis_summary.txt')
    with open(output_txt_path, 'w') as f:
        f.write(f"Wing Region Mean Intensity: {wing_mean_intensity:.2f}\n")
        f.write(f"Total Wingtip Pixels: {wingtip_total_pixels}\n")
        f.write(f"Darker Wingtip Pixels: {wingtip_darker_pixels_count}\n")
        f.write(f"Percentage of Wingtip Pixels Darker Than Wing Mean: {percentage_darker_pixels:.2f}%\n")
    
    print(f"\nResults saved to:")
    print(f"- {output_csv_path}")
    print(f"- {output_txt_path}")
    
    return wing_mean_intensity, percentage_darker_pixels

if __name__ == "__main__":
    analyze_wingtip_darkness()
