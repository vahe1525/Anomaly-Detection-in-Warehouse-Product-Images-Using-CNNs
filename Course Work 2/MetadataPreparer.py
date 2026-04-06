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

df = prepare_wood_metadata('data/wood')

# Ստուգում ենք արդյունքը
print("Առաջին 5 տողը.")
print(df.head())

print("\nԴեֆեկտների տեսակները և քանակը.")
print(sum(df[df['label'] == 1]['parent_folder'].value_counts()))
# print(df[df['label'] == 0]['parent_folder'].value_counts())

df.to_csv('wood_metadata.csv', index=False)
print("Metadata-ն պահպանվեց wood_metadata.csv ֆայլում:")


