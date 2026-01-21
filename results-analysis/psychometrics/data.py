import pandas as pd

study = "v3_prolific"
df = pd.read_csv(f'{study}_all_answers.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.columns = ['id', 'condition', 'component', 'instrument', 'handle', 'session_id', 'value']

nasa = df[df['instrument'] == "mot-NASA-TLX"]
sims = df[df['instrument'] == "mot-SIMS"]
tens = df[df['instrument'] == "mot-TENS"]
ues = df[df['instrument'] == "mot-UES"]
lp = df[df['instrument'] == "mot-LP"]

if __name__ == '__main__':
    pass
