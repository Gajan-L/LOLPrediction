import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def data_Process(file_dir):
    champs = pd.read_csv(file_dir+'champs.csv')
    matches = pd.read_csv(file_dir+'matches.csv')
    participants = pd.read_csv(file_dir+'participants.csv')
    stats1 = pd.read_csv(file_dir+'stats1.csv')
    stats2 = pd.read_csv(file_dir+'stats2.csv')
    stats = stats1.append(stats2)

    df = pd.merge(participants, stats, how='left', on=['id'], suffixes=('', '_y'))
    df = pd.merge(df, champs, how='left', left_on='championid', right_on='id', suffixes=('', '_y'))
    df = pd.merge(df, matches, how='left', left_on='matchid', right_on='id', suffixes=('', '_y'))

    def final_position(row):
        if row['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):
            return row['role']
        else:
            return row['position']

    df['adjposition'] = df.apply(final_position, axis=1)

    df['team'] = df['player'].apply(lambda x: '1' if x <= 5 else '2')
    df['team_role'] = df['team'] + ' - ' + df['adjposition']

    # remove matchid with duplicate roles, e.g. 3 MID in same team, etc
    remove_index = []
    for i in ('1 - MID', '1 - TOP', '1 - DUO_SUPPORT', '1 - DUO_CARRY', '1 - JUNGLE',
              '2 - MID', '2 - TOP', '2 - DUO_SUPPORT', '2 - DUO_CARRY', '2 - JUNGLE'):
        df_remove = df[df['team_role'] == i].groupby('matchid').agg({'team_role': 'count'})
        remove_index.extend(df_remove[df_remove['team_role'] != 1].index.values)

    # remove unclassified BOT, correct ones should be DUO_SUPPORT OR DUO_CARRY
    remove_index.extend(df[df['adjposition'] == 'BOT']['matchid'].unique())
    remove_index = list(set(remove_index))

    df = df[~df['matchid'].isin(remove_index)]

    df = df[['id', 'matchid', 'player', 'name', 'adjposition', 'team_role', 'win', 'kills', 'deaths', 'assists',
             'turretkills', 'totdmgtochamp', 'totheal', 'totminionskilled', 'goldspent', 'totdmgtaken', 'inhibkills',
             'pinksbought', 'wardsplaced', 'duration', 'platformid', 'seasonid', 'version']]

    return df