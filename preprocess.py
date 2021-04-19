from dataProcess import data_Process


# storing processed data for the label transferring of the game we want to predict
file_dir = './input/'
df = data_Process(file_dir)
df_4 = df[['matchid', 'player', 'name', 'team_role', 'win']]
df_4 = df_4.pivot(index='matchid', columns='team_role', values='name')
df_4 = df_4.reset_index()
df_4 = df_4.merge(df[df['player'] == 1][['matchid', 'win', 'platformid', 'seasonid', 'version']],
                      left_on='matchid', right_on='matchid', how='left')
df_4 = df_4[df_4.columns.difference(['matchid', 'version'])]
df_4 = df_4.rename(columns={'win': 'T1 win'})
df['name'].unique()
df_4 = df_4.dropna()
print(df_4)
df_4.to_csv('./DataFrame.csv')