import numpy as np
import pandas as pd
from gaitmap.example_data import get_healthy_example_imu_data_not_rotated
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import socket
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import itertools
import h5py
import os

np.random.seed(0)

def quaternion_multiply(q1, q0):
    w0, x0, y0, z0 = q0.T
    w1, x1, y1, z1 = np.transpose(q1)
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                       x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                      ],dtype=np.float64).T


example_dataset = get_healthy_example_imu_data_not_rotated()
station = socket.gethostname()
if station == 'Laptop_naas':
    project_path = 'C:\\Users\\Nassila\\Documents\\Projects\\Multiple_sclerosis\\Dual_task\\'#Raw_data\\test_data_H'
elif station == 'w10lloppici':
    project_path = 'D:\\Multiple_Sclerosis\\Gait_dual_task\\'#Raw_data\\test_data_H'
else:
    raise ValueError('No idea where the raw data is')

trials = [ifile for ifile in (Path(project_path)/'Raw_data//sensor_data').glob("*.csv")]

for trial in trials[:]:
    trial_name = trial.stem
    h5f = h5py.File(str(trial), 'r')
    quat = {}
    labels = {}
    rates = {}
    euler = {}
    for isensor in h5f['Processed'].keys():
        if h5f['Sensors'][isensor]['Configuration'].attrs['Label 0'].decode('UTF-8') in ['Sternum', 'Lumbar',  'Left Wrist', 'Right Wrist']:
            t = np.array(h5f['Sensors'][isensor]['Time'])
            rates[isensor] = h5f['Sensors'][isensor]['Configuration'].attrs['Sample Rate']
            labels[isensor] = h5f['Sensors'][isensor]['Configuration'].attrs['Label 0'].decode('UTF-8')

            q = np.array(h5f['Processed'][isensor]['Orientation'])
            quat[labels[isensor]] = q

            q_offset = [q[350, 0], -q[350, 1], -q[350, 2], -q[350, 3]]
            quat_calib = quaternion_multiply(q_offset, q)


            phi = np.arctan2(2*(quat_calib[:,0]*quat_calib[:,1] + quat_calib[:,2]*quat_calib[:,3]),
                            quat_calib[:,0]*quat_calib[:,0] - quat_calib[:,1]*quat_calib[:,1] - quat_calib[:,2]*quat_calib[:,2] + quat_calib[:,3]*quat_calib[:,3])

            theta = np.arcsin(-2*(quat_calib[:,1]*quat_calib[:,3] - quat_calib[:,0]*quat_calib[:,2]))
            psi = np.arctan2(2*(quat_calib[:,1]*quat_calib[:,2] + quat_calib[:,0]*quat_calib[:,3]),
                            quat_calib[:,0]*quat_calib[:,0] + quat_calib[:,1]*quat_calib[:,1] - quat_calib[:,2]*quat_calib[:,2] - quat_calib[:,3]*quat_calib[:,3])
            # theta = np.arcsin(R_yx / np.cos(phi))
            # psi = np.arcsin(R_zy / np.cos(phi))
            # r = R.from_quat(quat_calib)
            # euler[labels[isensor]] = np.column_stack(np.unwrap(r.as_euler('xyz', degrees=True), period=90))
            # euler[labels[isensor]] = np.rad2deg(np.column_stack([np.unwrap(phi), theta, psi]))
            euler[labels[isensor]] = np.rad2deg(np.column_stack([phi, theta, psi]))

        # euler[labels[isensor]] = np.rad2deg(np.column_stack((theta, phi, psi)))
    h5f.close()

    if len(set(rates.values())) != 1:
        raise Exception('Sensor rates are either different or missing')
    else:
        time_step = 1/list(rates.values())[0]

    trial_name = trial.stem
    new_path = Path(project_path) / 'sensors'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    column_names = [['time']] + [[ilabel + '/Angle_' + i for i in ['x', 'y', 'z']] for ilabel in list(labels.values())]
    column_names = [x for xs in column_names for x in xs]
    sensor_data = pd.DataFrame(data=np.zeros((euler[labels['10982']].shape[0], len(column_names))), columns=column_names)
    sensor_data.time = np.arange(0, euler[labels['10982']].shape[0], 1) * time_step

    for irot in list(euler.keys()):
        columns_to_fill = [irot + '/Angle_' + i for i in ['x', 'y', 'z']]
        sensor_data[columns_to_fill] = euler[irot]

    file_name = str(new_path) + '\\Angle_' + trial_name + '.csv'
    sensor_data.to_csv(file_name, index=False, sep='\t')

print('youpi')
