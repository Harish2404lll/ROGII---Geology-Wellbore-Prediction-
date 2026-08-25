import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import glob
import os

# 1. Define the folder
data_folder = "C:/Users/haris/Downloads/Harish/Hackathon/train/"

# 2. Let Python safely find the files instead of hardcoding the ID
search_path = os.path.join(data_folder, "*__horizontal_well.csv")
available_files = glob.glob(search_path)

if not available_files:
    print(f"❌ Error: Python could not find any files in: {data_folder}")
    print("Double check that the folder path is exactly correct.")
else:
    # Grab the very first file Python finds and extract its ID automatically
    hw_path = available_files[0]
    base_name = os.path.basename(hw_path)
    well_id = base_name.split('__')[0]
    
    # Automatically build the safe path for the Typewell
    tw_path = os.path.join(data_folder, f"{well_id}__typewell.csv")
    
    print(f"✅ Successfully found files for Well {well_id}!")
    
    # 3. Load the data
    hw_df = pd.read_csv(hw_path)
    tw_df = pd.read_csv(tw_path)
    
    # Extract the Gamma Ray signals as arrays (dropping blanks)
    # THE FIX IS HERE: .reshape(-1, 1) ensures the math engine gets the 1-D arrays it expects
    hw_gr_signal = hw_df['GR'].dropna().values.reshape(-1, 1)
    tw_gr_signal = tw_df['GR'].dropna().values.reshape(-1, 1)

    print(f"Horizontal Signal Length: {len(hw_gr_signal)}")
    print(f"Typewell Signal Length: {len(tw_gr_signal)}")

    # 4. Perform Dynamic Time Warping
    print("\nCalculating DTW Alignment... (This may take a few seconds)")
    distance, path = fastdtw(hw_gr_signal, tw_gr_signal, dist=euclidean)

    print(f"\n✅ DTW Complete!")
    print(f"Total Alignment Distance (Error): {distance:.2f}")

    # 5. Show how the algorithm matched the points
    print("\nFirst 5 alignment pairs (Horizontal Index -> Typewell Index):")
    for i in range(5):
        horizontal_idx = path[i][0]
        typewell_idx = path[i][1]
        print(f"Horizontal Row {horizontal_idx} matches Typewell Row {typewell_idx}")


    # ... (Your previous code ends here) ...

    # 6. Translate the DTW path into a new feature!
    print("\nExtracting the mapped TVT depths...")
    
    # We create a dictionary to store the mapping
    # If one horizontal row matches multiple typewell rows, we take the average
    dtw_mapping = {}
    for h_idx, t_idx in path:
        # Get the actual TVT value from the typewell at this index
        mapped_tvt = tw_df['TVT'].iloc[t_idx]
        
        if h_idx not in dtw_mapping:
            dtw_mapping[h_idx] = []
        dtw_mapping[h_idx].append(mapped_tvt)

    # Calculate the average mapped TVT for each horizontal row
    final_mapped_tvt = [np.mean(dtw_mapping[i]) for i in range(len(hw_gr_signal))]

    # 7. Add this as a brand new column to your horizontal dataframe!
    # Because we dropped NaN values earlier for the signal, we need to map these back carefully.
    # For this test, we'll create a cleaned version of the horizontal dataframe.
    hw_clean = hw_df.dropna(subset=['GR']).copy()
    hw_clean['DTW_TVT'] = final_mapped_tvt

    # Let's look at the final result!
    print("\n--- FINAL FEATURE EXTRACTION ---")
    print("Here is your horizontal data with the new DTW-calculated vertical depth:")
    display_columns = ['MD', 'GR', 'TVT', 'DTW_TVT']
    print(hw_clean[display_columns].head(10))