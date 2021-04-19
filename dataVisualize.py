import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataProcess import data_Process
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
plt.style.use('ggplot')

file_dir = './input/'
df = data_Process(file_dir)
df_v = df.copy()
# put upper and lower limit
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x < 30 else 30)
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x > 0 else 0)

plt.figure(figsize=(15, 10))
sns.violinplot(x="seasonid", y="wardsplaced", hue="win", data=df_v, split=True, inner='quartile')
plt.title('Win or Loss: Wardsplaced')
plt.show()


# put upper and lower limit
df_v['kills'] = df_v['kills'].apply(lambda x: x if x < 20 else 20)
df_v['kills'] = df_v['kills'].apply(lambda x: x if x > 0 else 0)

plt.figure(figsize=(15, 10))
sns.violinplot(x="seasonid", y="kills", hue="win", data=df_v, split=True, inner='quartile')
plt.title('Win or Loss: Kills')
plt.show()


df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr.corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize=(15, 10))
sns.heatmap(df_corr.corr(), cmap=cmap, annot=True, fmt='.2f', mask=mask, square=True, linewidths=.5, center=0)
plt.title('All factors (all games) and their correlation with winning')
plt.show()


df_corr_2 = df._get_numeric_data()
# for games less than 25 mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]
df_corr_2 = df_corr_2.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr_2.corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize=(15, 10))
sns.heatmap(df_corr_2.corr(), cmap=cmap, annot=True, fmt='.2f', mask=mask, square=True, linewidths=.5, center=0)
plt.title('All factors (game time less than 25min) and their correlation with winning')
plt.show()


df_corr_3 = df._get_numeric_data()
# for games more than 40 mins
df_corr_3 = df_corr_3[df_corr_3['duration'] > 2400]
df_corr_3 = df_corr_3.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr_3.corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize=(15, 10))
sns.heatmap(df_corr_3.corr(), cmap=cmap, annot=True, fmt='.2f', mask=mask, square=True, linewidths=.5, center=0)
plt.title('All factors (game time more than 40min) and their correlation with winning')
plt.show()


pd.options.display.float_format = '{:,.1f}'.format

df_win_rate = df.groupby('name').agg({'win': 'sum', 'name': 'count', 'kills': 'mean',
                                      'deaths': 'mean', 'assists': 'mean'})
df_win_rate.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win matches']/df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate', ascending=False)
df_win_rate = df_win_rate[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]
print('Top 10 win rate')
print(df_win_rate.head(10))
print('Bottom 10 win rate')
print(df_win_rate.tail(10))


df_win_rate.reset_index(inplace=True)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))


df_win_rate['color map'] = df_win_rate['win rate'].apply(lambda x: 'green' if x > 50 else 'red')
ax = df_win_rate.plot(kind='scatter', x='total matches', y='win rate', color=df_win_rate['color map'].tolist(),
                      figsize=(15, 10), title='Win rate of different champions with different numbers of matches')
label_point(df_win_rate['total matches'], df_win_rate['win rate'], df_win_rate['name'], ax)
plt.show()


pd.options.display.float_format = '{:,.1f}'.format

df_win_rate_role = df.groupby(['name', 'adjposition']).agg({'win': 'sum', 'name': 'count',
                                                            'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate_role.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate_role['win rate'] = df_win_rate_role['win matches']/df_win_rate_role['total matches'] * 100
df_win_rate_role['KDA'] = (df_win_rate_role['K'] + df_win_rate_role['A']) / df_win_rate_role['D']
df_win_rate_role = df_win_rate_role.sort_values('win rate', ascending=False)
df_win_rate_role = df_win_rate_role[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]

# occur > 0.01% of matches
df_win_rate_role = df_win_rate_role[df_win_rate_role['total matches'] > df_win_rate_role['total matches'].sum()*0.0001]
print('Top 10 win rate with role (occur > 0.01% of total # matches)')
print(df_win_rate_role.head(10))
print('Bottom 10 win rate with role (occur > 0.01% of total # matches)')
print(df_win_rate_role.tail(10))

df_2 = df.sort_values(['matchid', 'adjposition'], ascending=[1, 1])

df_2['shift 1'] = df_2['name'].shift()
df_2['shift -1'] = df_2['name'].shift(-1)

def get_matchup(x):
    if x['player'] <= 5:
        if x['name'] < x['shift -1']:
            name_return = x['name'] + ' vs ' + x['shift -1']
        else:
            name_return = x['shift -1'] + ' vs ' + x['name']
    else:
        if x['name'] < x['shift 1']:
            name_return = x['name'] + ' vs ' + x['shift 1']
        else:
            name_return = x['shift 1'] + ' vs ' + x['name']
    return name_return

df_2['match up'] = df_2.apply(get_matchup, axis = 1)
df_2['win_adj'] = df_2.apply(lambda x: x['win'] if x['name'] == x['match up'].split(' vs')[0] else 0, axis=1)

print(df_2.head(10))

df_matchup = df_2.groupby(['adjposition', 'match up']).agg({'win_adj': 'sum', 'match up': 'count'})
df_matchup.columns = ['win matches', 'total matches']
df_matchup['total matches'] = df_matchup['total matches'] / 2
df_matchup['win rate'] = df_matchup['win matches']/df_matchup['total matches']  * 100
df_matchup['dominant score'] = df_matchup['win rate'] - 50
df_matchup['dominant score (ND)'] = abs(df_matchup['dominant score'])
df_matchup = df_matchup[df_matchup['total matches'] > df_matchup['total matches'].sum()*0.0001]

df_matchup = df_matchup.sort_values('dominant score (ND)', ascending=False)
df_matchup = df_matchup[['total matches', 'dominant score']]
df_matchup = df_matchup.reset_index()

print('Dominant score +/- means first/second champion dominant:')

for i in df_matchup['adjposition'].unique():
        print('\n{}:'.format(i))
        print(df_matchup[df_matchup['adjposition'] == i].iloc[:,1:].head(5))

def get_best_counter(champion, role):
    df_matchup_temp = df_matchup[(df_matchup['match up'].str.contains(champion)) & (df_matchup['adjposition'] == role)]
    df_matchup_temp['champion'] = df_matchup_temp['match up'].apply(lambda x: x.split(' vs ')[0] if x.split(' vs ')[1] == champion else x.split(' vs ')[1])
    df_matchup_temp['advantage'] = df_matchup_temp.apply(lambda x: x['dominant score']*-1 if x['match up'].split(' vs ')[0] == champion else x['dominant score'], axis = 1)
    df_matchup_temp = df_matchup_temp[df_matchup_temp['advantage']>0].sort_values('advantage', ascending = False)
    print('Best counter for {} - {}:'.format(role, champion))
    print(df_matchup_temp[['champion', 'total matches', 'advantage']])
    return
