import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import socket
from datetime import date, datetime
import seaborn as sns
import matplotlib.pyplot as plt

station = socket.gethostname()
if station == 'Laptop_naas':
    project_path = 'C:\\Users\\Nassila\\Documents\\Projects\\Multiple_sclerosis\\Dual_task\\'
elif station == 'w10lloppici':
    project_path = 'D:\\Multiple_Sclerosis\\Gait_dual_task\\'#Raw_data\\test_data_H\\sensors\\20210430-083735_Walk.csv'
else:
    raise ValueError('No idea where the raw data is')

conf = pd.read_csv(project_path+'Raw_data\\conf.csv')
print(f'Initially, conf has {conf.shape[0]} sessions, with {len(set(conf.ID))} participants')
# Drop participant with EMS
conf = conf[~conf.ID.isin([447, 944, 56, 557, 1721, 1579, 844, 829])]
# Drop trials where nan is there for edss
conf_filt = conf[((conf.Group =='MS') & (conf.EDSS.notnull())) | (conf.Group =='HC')]
conf_no_EDSS = conf.merge(conf_filt.drop_duplicates(),
                   how='left', indicator=True)
conf_no_EDSS = conf_no_EDSS[conf_no_EDSS['_merge'] == 'left_only'].drop('_merge', axis=1)
copied_no_EDSS = 0
problematic_copies = []
for i, irow in conf_no_EDSS.iterrows():
    try:
        filename = f'P{irow.ID}_E{irow.EDSS}_S{irow.Session}_NG_{irow.File_NG[:-3]}.csv'
        shutil.copy(f"{project_path}Raw_data\\Sensor_data_fromMSZ\\{filename}",
                f"{project_path}Raw_data\\Sorted_data\\NO_EDSS\\{filename}"
                 )
        copied_no_EDSS += 1
    except Exception as e:
        problematic_copies.append([filename, str(e)])

    try:
        filename = f'P{irow.ID}_E{irow.EDSS}_S{irow.Session}_DT_{irow.File_DT[:-3]}.csv'
        shutil.copy(f"{project_path}Raw_data\\Sensor_data_fromMSZ\\{filename}",
                f"{project_path}Raw_data\\Sorted_data\\NO_EDSS\\{filename}"
                    )
        copied_no_EDSS += 1
    except Exception as e:
        problematic_copies.append([filename, str(e)])

print(f'After dropping missing EDSS, conf has {conf_filt.shape[0]} sessions, '
      f'with {len(set(conf_filt.ID))} participants')
print(f'Copied {copied_no_EDSS} files, but had issues copying {len(problematic_copies)} files')

copied_type_off = 0
copied_ok = 0
problematic_type_off = 0
problematic_ok = 0
for i, irow in conf_filt.iterrows():
    if irow.Group == 'MS' and irow.MS_type not in ['PPMS', 'RRMS', 'SPMS']:
        try:
            filename = f'P{irow.ID}_E{irow.EDSS}_S{irow.Session}_NG_{irow.File_NG[:-3]}.csv'
            shutil.copy(f"{project_path}Raw_data\\Sensor_data_fromMSZ\\{filename}",
                    f"{project_path}Raw_data\\Sorted_data\\Diff_Type\\{filename}"
                     )
            copied_type_off += 1
        except Exception as e:
            problematic_copies.append([filename, str(e)])
            problematic_type_off +=1

        try:
            filename = f'P{irow.ID}_E{irow.EDSS}_S{irow.Session}_DT_{irow.File_DT[:-3]}.csv'
            shutil.copy(f"{project_path}Raw_data\\Sensor_data_fromMSZ\\{filename}",
                    f"{project_path}Raw_data\\Sorted_data\\Diff_Type\\{filename}"
                        )
            copied_type_off += 1
        except Exception as e:
            problematic_copies.append([filename, str(e)])
            problematic_type_off += 1
    else:
        try:
            filename = f'P{irow.ID}_E{irow.EDSS}_S{irow.Session}_NG_{irow.File_NG[:-3]}.csv'
            shutil.copy(f"{project_path}Raw_data\\Sensor_data_fromMSZ\\{filename}",
                    f"{project_path}Raw_data\\Sorted_data\\To_process\\{filename}"
                     )
            copied_ok += 1
        except Exception as e:
            problematic_copies.append([filename, str(e)])
            problematic_ok += 1

        try:
            filename = f'P{irow.ID}_E{irow.EDSS}_S{irow.Session}_DT_{irow.File_DT[:-3]}.csv'
            shutil.copy(f"{project_path}Raw_data\\Sensor_data_fromMSZ\\{filename}",
                    f"{project_path}Raw_data\\Sorted_data\\To_process\\{filename}"
                        )
            copied_ok += 1
        except Exception as e:
            problematic_copies.append([filename, str(e)])
            problematic_ok += 1

conf_filt = conf_filt[((conf_filt.Group=='MS') & (conf_filt.MS_type.isin(['PPMS', 'RRMS', 'SPMS'])))
                      | (conf_filt.Group == 'HC')]
print(f'After keeping only main three types, conf has {conf_filt.shape[0]} sessions, '
      f'with {len(set(conf_filt.ID))} participants')

for i, ie in problematic_copies[8:]:
    to_drop = conf_filt[conf_filt.loc[:,f"File_{i.split('_')[3]}"].str.startswith(i.split('_')[4])].index
    if to_drop.shape[0]!=0:
        conf_filt.drop(to_drop, inplace=True)

conf_filt.reset_index(drop=True, inplace=True)

# occurence = 1
# par_id = 0
# for index, row in conf_filt.iterrows():
#     if row.ID==par_id:
#         row.Session = occurence
#     else:


# Find time between session
participants_w_only_one = []
time_delay = pd.DataFrame(columns = ['ID', 'MS_type', 'diff_1', 'diff_2', 'diff_3', 'diff_4', 'diff_5'])

for ipar in set(conf_filt.ID):
    conf_par = conf_filt[conf_filt.ID==ipar]
    conf_filt.loc[conf_par.index, 'Session'] = np.arange(1, conf_par.shape[0]+1, 1)

    if conf_par.shape[0] == 1:
        participants_w_only_one.append(ipar)
        time_delay = pd.concat([time_delay, pd.DataFrame(
            {'ID': [ipar],
             'MS_type': [conf_par.MS_type.values[0]],
             'diff_1': -10,
             'diff_2': -10,
             'diff_3': -10,
             'diff_4': -10,
             'diff_5': -10,}
        )], ignore_index=True)
    else:
        session_dates = [datetime.strptime(i, "%Y-%m-%d") for i in conf_par.Date_gait]
        delta = [session_dates[i]-session_dates[i-1] for i in range(1, len(session_dates), 1)]

        new_row = pd.DataFrame(
            {'ID': [ipar],
             'MS_type': [conf_par.MS_type.values[0]],
             'diff_1': -10,
             'diff_2': -10,
             'diff_3': -10,
             'diff_4': -10,
             'diff_5': -10, }
        )
        for i, idiff in enumerate(['diff_1', 'diff_2', 'diff_3', 'diff_4', 'diff_5'][:len(delta)]):
            new_row[idiff] = delta[i].days
        time_delay = pd.concat([time_delay, new_row], ignore_index=True)

df = pd.melt(time_delay, ['ID', 'MS_type'], value_name='Difference')
df = df[df.Difference>0]
df = df[df.MS_type.isin(['PPMS', 'SPMS', 'RRMS'])]
# sns.relplot(data=df, x='ID', y='Difference', hue='variable', col='MS_type',
#             kind='scatter')
g = sns.relplot(data=df[df.MS_type.notnull()], x='ID', y='Difference',
                row='MS_type', col='variable',
                hue='MS_type',
                kind='scatter')
(g.map(plt.axhline, y=365, color=".8", dashes=(2, 1), zorder=10)
  .map(plt.axhline, y=730, color=".8", dashes=(2, 1), zorder=9)
  .map(plt.axhline, y=182, color=".8", dashes=(2, 1), zorder=8)
  .map(plt.axhline, y=548, color=".8", dashes=(2, 1), zorder=11)
  .set_axis_labels("Participant IDs", "Days between two successive sessions")
  .set_titles("Region: {col_name} sessions"))

[i.set_title('') for i in g.axes.flatten()[5:]]
[g.fig.delaxes(ax) for ax in g.axes.flatten() if not ax.has_data()]

plt.savefig(project_path+'Raw_data\\Time_btw_sessions.svg')
conf_filt.to_csv(project_path+'Raw_data\\conf_filt.csv', sep='\t')
print('Let go')
