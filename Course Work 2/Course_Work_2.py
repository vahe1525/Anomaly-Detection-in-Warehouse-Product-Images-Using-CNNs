print("------1")

import pandas as pd
from sklearn.model_selection import train_test_split
print("1")
df = pd.read_csv('wood_metadata.csv')

# Նախ բաժանում ենք ընդհանուր դատասեթը 80% Train և 20% Test
train_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    stratify=df['label'], # Սա ամենակարևոր մասն է
    random_state=42
)

print("2")
# Հետո Train-ի մի մասը (օրինակ 15%-ը) վերցնում ենք որպես Validation
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.10, 
    stratify=train_df['label'], 
    random_state=42
)

print(f"Մարզման նկարներ (Train): {len(train_df)}")
print(f"Ստուգման նկարներ (Val): {len(val_df)}")
print(f"Թեստավորման նկարներ (Test): {len(test_df)}")

print("------1")
# Ստուգենք անոմալիաների բաշխումը Test-ում
print("\nԱնոմալիաների քանակը Test set-ում.")
print(test_df['label'].value_counts())
