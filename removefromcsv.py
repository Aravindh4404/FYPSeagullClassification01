import pandas as pd
import re
import os


def load_and_clean_csv(csv_file_path, output_file_path=None):
    """
    Load CSV file and delete rows containing specified image names

    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str): Path for the cleaned CSV file (optional)

    Returns:
        pd.DataFrame: Cleaned dataframe
        dict: Summary of deletions
    """

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None

    # Store original count
    original_count = len(df)

    # Define image names to delete
    slaty_backed_images = [
        "004", "016", "025", "042", "046", "102", "108", "109", "112",
        "0H5A6244", "0H5A6657", "0H5A9971", "3O4A1145 2", "3O4A1687", "3O4A3053", "3O4A3226",
        "3O4A5274 2", "3O4A6909", "3O4A7515 2", "3O4A8300", "3O4A8923", "480 (14)", "4D2A6625",
        "4D2A6730", "4D2A6769", "4D2A7037", "4D2A7433", "FS7E2526", "FS7E2560"
    ]

    glaucous_winged_images = [
        "12", "14", "19", "20", "26", "28", "30", "52", "53", "55", "58",
        "64", "67", "70", "72", "75", "95", "105", "106", "116", "117",
        "118", "121", "123", "125", "130", "138", "0Z2A1082", "0Z2A1461", "0Z2A2497",
        "0Z2A8458 (1)", "0Z2A9154", "tmpbmib4ve2"
    ]

    # Combine all images to delete
    all_images_to_delete = slaty_backed_images + glaucous_winged_images

    # Track deletions
    deletion_summary = {
        'slaty_backed_deleted': [],
        'glaucous_winged_deleted': [],
        'not_found': [],
        'total_deleted': 0
    }

    # Function to check if image name matches any deletion criteria
    def should_delete_row(image_name):
        if pd.isna(image_name):
            return False

        image_name_str = str(image_name).strip()

        # Direct match
        for delete_name in all_images_to_delete:
            # Exact match
            if image_name_str == delete_name:
                return True

            # Check if delete_name is contained in image_name (for cases like AH5A6244)
            if delete_name in image_name_str:
                return True

            # Check if image_name contains delete_name (for partial matches)
            if delete_name.replace('(', '').replace(')', '') in image_name_str:
                return True

            # Special handling for AZ2A series
            if delete_name in ['1082', '1461', '2497', '8458(1)', '9154']:
                if f"A{delete_name.replace('(1)', '')}" in image_name_str or f"AZ2A{delete_name.replace('(1)', '')}" in image_name_str:
                    return True

        return False

    # Create a mask for rows to keep
    mask_to_keep = ~df['image_name'].apply(should_delete_row)

    # Get rows that will be deleted for summary
    rows_to_delete = df[~mask_to_keep]

    # Apply the mask to keep only non-matching rows
    df_cleaned = df[mask_to_keep].copy()

    # Generate deletion summary
    for _, row in rows_to_delete.iterrows():
        image_name = str(row['image_name']).strip()
        species = str(row.get('species', 'Unknown')).strip()

        if any(img in image_name for img in slaty_backed_images):
            deletion_summary['slaty_backed_deleted'].append(image_name)
        elif any(img in image_name for img in glaucous_winged_images):
            deletion_summary['glaucous_winged_deleted'].append(image_name)

    deletion_summary['total_deleted'] = len(rows_to_delete)

    # Check for images that weren't found
    found_images = set()
    for _, row in rows_to_delete.iterrows():
        image_name = str(row['image_name']).strip()
        for delete_name in all_images_to_delete:
            if delete_name in image_name or image_name in delete_name:
                found_images.add(delete_name)

    deletion_summary['not_found'] = [img for img in all_images_to_delete if img not in found_images]

    # Print summary
    print(f"\n=== DELETION SUMMARY ===")
    print(f"Original rows: {original_count}")
    print(f"Rows deleted: {deletion_summary['total_deleted']}")
    print(f"Remaining rows: {len(df_cleaned)}")

    print(f"\nSlaty-backed images deleted ({len(deletion_summary['slaty_backed_deleted'])}):")
    for img in deletion_summary['slaty_backed_deleted']:
        print(f"  - {img}")

    print(f"\nGlaucous-winged images deleted ({len(deletion_summary['glaucous_winged_deleted'])}):")
    for img in deletion_summary['glaucous_winged_deleted']:
        print(f"  - {img}")

    if deletion_summary['not_found']:
        print(f"\nImages from deletion list not found in CSV ({len(deletion_summary['not_found'])}):")
        for img in deletion_summary['not_found']:
            print(f"  - {img}")

    # Save cleaned CSV
    if output_file_path is None:
        base_name = os.path.splitext(csv_file_path)[0]
        output_file_path = f"{base_name}_cleaned.csv"

    try:
        df_cleaned.to_csv(output_file_path, index=False)
        print(f"\nCleaned CSV saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving cleaned CSV: {e}")

    return df_cleaned, deletion_summary


def main():
    """
    Main function to run the CSV cleaning process
    """
    # Specify your CSV file path here
    csv_file_path = "C:/Users/aravi/Desktop/dark_pixel_results_all_images.csv"  # Update this path as needed

    # Optional: specify output file path
    output_file_path = "dark_pixel_results_all_images_cleaned.csv"

    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found!")
        print("Please update the csv_file_path variable with the correct path to your CSV file.")
        return

    # Run the cleaning process
    cleaned_df, summary = load_and_clean_csv(csv_file_path, output_file_path)

    if cleaned_df is not None:
        print(f"\n=== SUCCESS ===")
        print(f"Process completed successfully!")
        print(f"Original file: {csv_file_path}")
        print(f"Cleaned file: {output_file_path}")

        # Optional: Display first few rows of cleaned data
        print(f"\nFirst 5 rows of cleaned data:")
        print(cleaned_df.head())

        # Optional: Show species distribution
        if 'species' in cleaned_df.columns:
            print(f"\nSpecies distribution in cleaned data:")
            print(cleaned_df['species'].value_counts())


if __name__ == "__main__":
    main()