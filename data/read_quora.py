import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./quora_duplicate_questions.tsv', delimiter='\t')
df_duplicates = df[df['is_duplicate'] == 1]
train_data, test_data = train_test_split(df_duplicates, train_size=0.9, random_state=1234)

print(f"Length of train data: {len(train_data)}")
print(f"Length of test data: {len(test_data)}")

train_data.to_csv('quora/train.csv')
test_data.to_csv('quora/test.csv')
