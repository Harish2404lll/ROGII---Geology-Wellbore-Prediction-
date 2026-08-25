import pandas as pd


# Load the data
try:
    hw_df = pd.read_csv('0a57a29c__horizontal_well.csv')
    tw_df = pd.read_csv('0a57a29c__typewell.csv')

    # Basic summaries
    hw_info = {
        'shape': hw_df.shape,
        'missing_tvt': hw_df['TVT'].isnull().sum(),
        'missing_tvt_input': hw_df['TVT_input'].isnull().sum(),
        'gr_min': hw_df['GR'].min(),
        'gr_max': hw_df['GR'].max(),
        'md_min': hw_df['MD'].min(),
        'md_max': hw_df['MD'].max()
    }
    
    tw_info = {
        'shape': tw_df.shape,
        'gr_min': tw_df['GR'].min(),
        'gr_max': tw_df['GR'].max(),
        'tvt_min': tw_df['TVT'].min(),
        'tvt_max': tw_df['TVT'].max(),
        'formations': tw_df['Geology'].unique().tolist()
    }
    
    print("Horizontal Well Info:", hw_info)
    print("Typewell Info:", tw_info)
    
except Exception as e:
    print("Error:", e)