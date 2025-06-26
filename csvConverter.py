import pandas as pd

def convert_torcs_csv(input_file, output_file):
    # Load the complete dataset
    df = pd.read_csv(input_file)
    
    # Mapping between your desired names and original column names
    column_mapping = {
        'ACCELERATION': 'accel',
        'BRAKE': 'brake',
        'STEERING': 'steer',
        'SPEED': 'speedX',
        'TRACK_POSITION': 'trackPos',
        'ANGLE_TO_TRACK_AXIS': 'angle',
        **{f'TRACK_EDGE_{i}': f'track{i+1}' for i in range(18)}
    }
    
    # Create new DataFrame with only the columns we want
    standardized_df = pd.DataFrame()
    
    for new_name, original_name in column_mapping.items():
        if original_name in df.columns:
            standardized_df[new_name] = df[original_name]
        else:
            print(f"Warning: Column {original_name} not found, filling with zeros")
            standardized_df[new_name] = 0
    
    # Save to new CSV
    standardized_df.to_csv(output_file, index=False)
    print(f"Successfully created standardized file: {output_file}")

# Usage example
convert_torcs_csv("1.csv", "standardized_torcs_data.csv")
