import pandas as pd
from pathlib import Path

def prepare_wood_metadata(root_path):
    data = []
    root = Path(root_path)
    
    for img_path in root.rglob('*.png'):
        
        if 'ground_truth' in img_path.parts:
            continue
            
        parent_folder = img_path.parent.name
        
        if parent_folder == 'good':
            label = 0 
        else:
            label = 1 
            
        data.append({
            'image_path': str(img_path),
            'parent_folder': parent_folder,
            'label': label
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    # This block only runs when you execute MetadataPreparer.py directly.
    # It does NOT run when main.py imports prepare_wood_metadata from this file.
    df = prepare_wood_metadata('data/wood')

    print("First 5 rows:")
    print(df.head())
    print(f"\nTotal anomaly images: {sum(df[df['label'] == 1]['parent_folder'].value_counts())}")

    df.to_csv('wood_metadata.csv', index=False)
    print("\nSaved: wood_metadata.csv")


