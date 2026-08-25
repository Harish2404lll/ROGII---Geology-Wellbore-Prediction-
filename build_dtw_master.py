import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import glob
import os

# 1. Define your folder
data_folder = "C:/Users/haris/Downloads/Harish/Hackathon/train/"

# Find all the horizontal well files
search_path = os.path.join(data_folder, "*__horizontal_well.csv")
horizontal_files = glob.glob(search_path)

all_processed_wells = []
total_wells = len(horizontal_files)

print(f"Starting DTW processing for {total_wells} wells...")
print("This will take a while. Please do not close VS Code!\n")

# 2. Loop through every single well
for index, hw_path in enumerate(horizontal_files):
    try:
        # Get the ID and find the matching Typewell
        base_name = os.path.basename(hw_path)
        well_id = base_name.split('__')[0]
        tw_path = os.path.join(data_folder, f"{well_id}__typewell.csv")
        
        # Load both files
        hw_df = pd.read_csv(hw_path)
        tw_df = pd.read_csv(tw_path)
        
        # Add the WELL_ID column so we don't lose it!
        hw_df['WELL_ID'] = well_id
        
        # Find exactly which rows have valid Gamma Ray data
        valid_h_idx = hw_df['GR'].dropna().index
        valid_t_idx = tw_df['GR'].dropna().index
        
        # Extract the signals and format them for SciPy
        hw_gr_signal = hw_df.loc[valid_h_idx, 'GR'].values.reshape(-1, 1)
        tw_gr_signal = tw_df.loc[valid_t_idx, 'GR'].values.reshape(-1, 1)
        
        # Run the DTW algorithm for this specific pair
        distance, path = fastdtw(hw_gr_signal, tw_gr_signal, dist=euclidean)
        
        # Map the vertical TVT depths back to the horizontal rows
        dtw_mapping = {}
        for h_step, t_step in path:
            orig_h_idx = valid_h_idx[h_step]
            orig_t_idx = valid_t_idx[t_step]
            
            mapped_tvt = tw_df.loc[orig_t_idx, 'TVT']
            
            if orig_h_idx not in dtw_mapping:
                dtw_mapping[orig_h_idx] = []
            dtw_mapping[orig_h_idx].append(mapped_tvt)
            
        # Create the brand new feature column (default to empty/NaN)
        hw_df['DTW_TVT'] = np.nan
        
        # Fill the column with the calculated depths
        for row_idx, tvt_list in dtw_mapping.items():
            hw_df.loc[row_idx, 'DTW_TVT'] = np.mean(tvt_list)
            
        # Add this finished well to our master list
        all_processed_wells.append(hw_df)
        
        # Print progress every 50 wells so you know it's working
        if (index + 1) % 50 == 0 or (index + 1) == total_wells:
            print(f"✅ Processed {index + 1} out of {total_wells} wells...")
            
    except Exception as e:
        print(f"❌ Skipped Well {well_id} due to error: {e}")

# 3. Stitch them all together into the new Master File
print("\nStitching all 773 processed wells into one massive dataset...")
master_v2_df = pd.concat(all_processed_wells, ignore_index=True)

output_filename = os.path.join(data_folder, 'MASTER_training_data_v2.csv')
master_v2_df.to_csv(output_filename, index=False)

print("\n🎉 SUCCESS! Pipeline Phase 1 Complete.")
print(f"Saved your upgraded dataset to: {output_filename}")