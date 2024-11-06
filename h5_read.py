import h5py
# import opensim as osim
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import socket

station = socket.gethostname()
if station == 'Laptop_naas':
    project_path = 'C:\\Users\\Nassila\\Documents\\Projects\\Multiple_sclerosis\\Dual_task\\'
elif station == 'w10lloppici':
    project_path = 'D:\\Multiple_Sclerosis\\Gait_dual_task\\'
else:
    raise ValueError('No idea where the raw data is')
# project_path= 'D:\Multiple_Sclerosis\Gait_dual_task'
# mobilityLab_settings = osim.APDMDataReaderSettings(f"{project_path}/results/_templates/IMU_Mappings.xml")
# mobLab = osim.APDMDataReader(mobilityLab_settings)

trials = [ifile for ifile in (Path(project_path) / 'Raw_data/test_data_H').glob("*.H5") if ifile.stem.endswith('Walk')]
save_acc_gyro = True

for trial in trials[-1:]:
    h5f = h5py.File(str(trial), 'r')
    quat = {}
    acc = {}
    gyro = {}
    labels = {}
    rates = {}
    euler = {}
    for isensor in h5f['Processed'].keys():
        t = np.array(h5f['Sensors'][isensor]['Time'])
        rates[isensor] = h5f['Sensors'][isensor]['Configuration'].attrs['Sample Rate']
        labels[isensor] = h5f['Sensors'][isensor]['Configuration'].attrs['Label 0'].decode('UTF-8')
        acc[isensor] = np.array(h5f['Sensors'][isensor]['Accelerometer'])
        gyro[isensor] = np.array(h5f['Sensors'][isensor]['Gyroscope'])

        q =  np.array(h5f['Processed'][isensor]['Orientation'])
        quat[isensor] = q

        R_zx = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        R_yx = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        R_zy = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])

        phi = -np.arcsin(R_zx)
        theta = np.arcsin(R_yx / np.cos(phi))
        psi = np.arcsin(R_zy / np.cos(phi))

        euler[isensor] = np.rad2deg(np.column_stack((theta, phi, psi)))
    h5f.close()

    if len(set(rates.values())) != 1:
        raise Exception('Sensor rates are either different or missing')
    else:
        time_step = 1/list(rates.values())[0]

    if save_acc_gyro:
        trial_name = trial.stem
        new_path = Path(project_path) / 'Raw_data/test_data_H/sensors'
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        column_names = [['time']] + [[ilabel + '/acc_' + i for i in ['x', 'y', 'z']] + [
            ilabel + '/gyr_' + i for i in ['x', 'y', 'z']] for ilabel in list(labels.values())]
        column_names = [x for xs in column_names for x in xs]
        sensor_data = pd.DataFrame(data=np.zeros((acc[isensor].shape[0], len(column_names))), columns=column_names)
        sensor_data.time = np.arange(0,acc['10982'].shape[0],1) * time_step

        for iacc in list(acc.keys()):
            ilabel = labels[iacc]
            columns_to_fill = [ilabel + '/acc_' + i for i in ['x', 'y', 'z']]
            sensor_data[columns_to_fill] = acc[iacc]

            columns_to_fill = [ilabel + '/gyr_' + i for i in ['x', 'y', 'z']]
            sensor_data[columns_to_fill] = gyro[iacc]

        file_name = str(new_path)+'\\'+trial_name+'.csv'
        sensor_data.to_csv(file_name, index=False, sep='\t')


    data = [pd.DataFrame({'Time': np.arange(0,acc['10982'].shape[0],1) * time_step})]*3

    data[0] = [data[0].join(pd.DataFrame(data=isen, columns=['X', 'Y', 'Z'])).assign(sensor=list(labels.values())[
        i]).assign(which='Acc')  for i, isen in enumerate(acc.values())]
    data[1] = [data[1].join(pd.DataFrame(data=isen, columns=['X', 'Y', 'Z'])).assign(sensor=list(labels.values())[
        i]).assign(which='Gyro') for i, isen in enumerate(gyro.values())]
    data[2] = [data[2].join(pd.DataFrame(data=isen, columns=['X', 'Y', 'Z'])).assign(sensor=list(labels.values())[
        i]).assign(which='Euler') for i, isen in enumerate(euler.values())]

    df = pd.concat([pd.concat(data[0]), pd.concat(data[1]), pd.concat(data[2])])
    df = df.melt(['Time', 'which', 'sensor'])
    sns.relplot(data=df, x='Time',y='value', hue='variable', col='which', row='sensor', kind='line', facet_kws={'sharey': False, 'sharex': True})





    if not all([np.array_equal(list(time.values())[0], arr) for arr in list(time.values())]) or list(time.values())[
        0].shape[0] != list(quat.values())[0].shape[0]:
        raise Exception('Time and quaternions shape do not match')






    table = mobLab.read(str(trial))
    trial_name = trial.stem
    quatTable = mobLab.getOrientationsTable(table)
    osim.STOFileAdapterQuaternion.write(quatTable, trial_name +'_orientations.sto')