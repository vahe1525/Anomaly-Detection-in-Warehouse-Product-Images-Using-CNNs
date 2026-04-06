import pandas as pd
from pathlib import Path

def prepare_wood_metadata(root_path):
    data = []
    root = Path(root_path)
    
    # Փնտրում ենք բոլոր .png ֆայլերը բոլոր ենթաթղթապանակներում
    for img_path in root.rglob('*.png'):
        
        # Բացառում ենք ground_truth թղթապանակը (այնտեղ անոմալիաների մասկաներն են)
        if 'ground_truth' in img_path.parts:
            continue
            
        # Հիմնական տրամաբանությունը.
        # img_path.parent.name-ը տալիս է այն թղթապանակի անունը, որտեղ նկարն է
        parent_folder = img_path.parent.name
        
        if parent_folder == 'good':
            label = 0  # Նորմալ ապրանք
        else:
            label = 1  # Անոմալիա (քանի որ անունը 'good' չէ, այլ օր. 'scratch' կամ 'hole')
            
        data.append({
            'image_path': str(img_path),
            'parent_folder': parent_folder, # Սա պահում ենք ստուգման համար
            'label': label
        })
    
    return pd.DataFrame(data)

# Օգտագործումը
df = prepare_wood_metadata('data/wood')

# Ստուգում ենք արդյունքը
print("Առաջին 5 տողը.")
print(df.head())

print("\nԴեֆեկտների տեսակները և քանակը.")
print(sum(df[df['label'] == 1]['parent_folder'].value_counts()))
# print(df[df['label'] == 0]['parent_folder'].value_counts())

# 1. Պահպանում ենք CSV ֆայլը
df.to_csv('wood_metadata.csv', index=False)
print("Metadata-ն պահպանվեց wood_metadata.csv ֆայլում:")


