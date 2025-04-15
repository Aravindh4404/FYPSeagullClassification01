import os
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from Features_Analysis.Intensity_New.wing_intensity_analysis import main as wing_intensity_main
from Features_Analysis.Intensity_New.wingtip_darkness_analysis import main as wingtip_darkness_main

def main():
    """
    Run the complete intensity analysis pipeline:
    1. Wing intensity analysis
    2. Wingtip darkness analysis
    """
    print("=" * 80)
    print("STARTING WING INTENSITY ANALYSIS")
    print("=" * 80)
    wing_intensity_main()
    
    print("\n" + "=" * 80)
    print("STARTING WINGTIP DARKNESS ANALYSIS")
    print("=" * 80)
    wingtip_darkness_main()
    
    print("\n" + "=" * 80)
    print("INTENSITY ANALYSIS COMPLETE")
    print("=" * 80)
    print("Results are saved in:")
    print("- Wing_Intensity_Results_New/")
    print("- Wingtip_Darkness_Results_New/")

if __name__ == "__main__":
    main() 