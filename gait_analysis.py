import numpy as np
import pandas as pd
from gaitmap.example_data import get_healthy_example_imu_data_not_rotated
from gaitmap.preprocessing import sensor_alignment
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.utils.rotations import flip_dataset, rotation_from_angle
from gaitmap.event_detection import RamppEventDetection
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory, RegionLevelTrajectory, RtsKalman
from gaitmap.parameters import TemporalParameterCalculation
from gaitmap.parameters import SpatialParameterCalculation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import socket
import itertools

np.random.seed(0)

example_dataset = get_healthy_example_imu_data_not_rotated()
station = socket.gethostname()
if station == 'Laptop_naas':
    path_data = 'C:\\Users\\Nassila\\Documents\\Projects\\Multiple_sclerosis\\Dual_task\\Raw_data\\test_data_H\\sensors\\20210430-083735_Walk.csv'
elif station == 'w10lloppici':
    path_data = 'D:\\Multiple_Sclerosis\\Gait_dual_task\\Raw_data\\test_data_H\\sensors\\20210430-083735_Walk.csv'
else:
    raise ValueError('No idea where the raw data is')

data_set = pd.read_csv(path_data, sep='\t')
sampling_rate_hz = 1/data_set.time[1] #204.8
data_set_foot = data_set[[icol for icol in data_set.columns if 'Foot' in icol or icol=='time']]
time = data_set_foot.time.values
dst = data_set_foot.set_index('time')
dst.columns = pd.MultiIndex.from_arrays(np.transpose([i.split('/') for i in dst.columns]),
                                           names=['sensor', 'axis'])
dst.loc[:, pd.IndexSlice[:, ['gyr_x', 'gyr_y', 'gyr_z']]] *= 180/np.pi

# # rotate left_sensor first by -90 deg around the x-axis, followed by a -90 deg rotation around the z-axis
# left_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90)) * rotation_from_angle(
#     np.array([0, 0, 1]), np.deg2rad(-90)
# )
#
# # rotate right_sensor first by +90 deg around the x-axis, followed by a +90 deg rotation around the z-axis
# right_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(90)) * rotation_from_angle(
#     np.array([0, 0, 1]), np.deg2rad(90)
# )

# rotations = dict(left_sensor=left_rot, right_sensor=right_rot)
# dataset_sf = flip_dataset(example_dataset, rotations)

dataset_sf_al_to_gr = sensor_alignment.align_dataset_to_gravity(
    dataset=dst,
    sampling_rate_hz=sampling_rate_hz,
    window_length_s=1.0,
    static_signal_th=5
)

# --------------------- Stride Segmentation
dtw = BarthDtw( max_cost=3.8, min_match_length_s=0.7, max_match_length_s=1.3)
# Convert data to foot-frame
bf_data = convert_to_fbf(dataset_sf_al_to_gr, left_like="Left", right_like="Right")
dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

# Check detected strides
fig, axs = plt.subplots(nrows=3, ncols=2,  sharex=True, figsize=(10, 5))
foot_laterality = ["Left Foot", "Right Foot"]
for i, sensor in enumerate(foot_laterality):
    axs[0, i].plot(time, dtw.data[sensor]["gyr_ml"].reset_index(drop=True))
    axs[0, i].set_ylabel("gyro [deg/s]")
    axs[1, i].plot(time, dtw.cost_function_[sensor])
    axs[1, i].set_ylabel("dtw cost [a.u.]")
    axs[1, i].axhline(dtw.max_cost, color="k", linestyle="--")
    axs[2, i].imshow(dtw.acc_cost_mat_[sensor], aspect="auto", extent=[0, time[-1], 120, 0])
    axs[2, i].set_ylabel("template position [#]")
    for p in dtw.paths_[sensor]:
        axs[2, i].plot(p.T[1]/sampling_rate_hz, p.T[0])
    for s in dtw.matches_start_end_original_[sensor]:
        axs[1, i].axvspan(*s/sampling_rate_hz, alpha=0.3, color="g")
    for _, s in dtw.stride_list_[sensor][["start", "end"]].iterrows():
        axs[0, i].axvspan(*s/sampling_rate_hz, alpha=0.3, color="g")

    axs[2, i].set_xlabel("time [s]")
    axs[0, i].set_title(sensor)
fig.tight_layout()
plt.show()

# --------------------- Event detection
ed = RamppEventDetection()
ed = ed.detect(data=bf_data, stride_list=dtw.stride_list_, sampling_rate_hz=sampling_rate_hz)

# Check detected events
fig, [[ax1l, ax1r], [ax2l, ax2r]] = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))
ax1l.plot(time, bf_data.reset_index(drop=True)["Left Foot"][["gyr_ml"]])
ax2l.plot(time, bf_data.reset_index(drop=True)["Left Foot"][["acc_pa"]])
ax1r.plot(time, bf_data.reset_index(drop=True)["Right Foot"][["gyr_ml"]])
ax2r.plot(time, bf_data.reset_index(drop=True)["Right Foot"][["acc_pa"]])

ic_idx= [ed.min_vel_event_list_[k]["ic"].to_numpy().astype(int) for k in ["Left Foot", "Right Foot"]]
tc_idx = [ed.min_vel_event_list_[k]["tc"].to_numpy().astype(int) for k in ["Left Foot", "Right Foot"]]
min_vel_idx = [ed.min_vel_event_list_[k]["min_vel"].to_numpy().astype(int) for k in ["Left Foot", "Right Foot"]]

for iside, side in enumerate([[ax1l, ax2l], [ax1r, ax2r]]):
    for ax, sensor in zip(side, ["gyr_ml", "acc_pa"]):
        for i, stride in ed.min_vel_event_list_[foot_laterality[iside]].iterrows():
            ax.axvline(stride["start"]/sampling_rate_hz, color="g")
            ax.axvline(stride["end"]/sampling_rate_hz, color="r")

        ax.scatter(ic_idx[iside]/sampling_rate_hz, bf_data[foot_laterality[iside]][sensor].to_numpy()[ic_idx[iside]],marker="*",s=100,
                   color="r", zorder=3, label="ic",
        )

        ax.scatter(tc_idx[iside]/sampling_rate_hz, bf_data[foot_laterality[iside]][sensor].to_numpy()[tc_idx[iside]], marker="p", s=50,
                   color="g", zorder=3, label="tc",
                   )

        ax.scatter(min_vel_idx[iside]/sampling_rate_hz, bf_data[foot_laterality[iside]][sensor].to_numpy()[min_vel_idx[iside]],
                   marker="s", s=50,
                   color="y", zorder=3, label="min_vel",
                   )

        ax.grid(True)
    side[0].set_title(f"Events of min_vel strides for the {foot_laterality[iside]}")
ax1l.set_ylabel("gyr_ml (°/s)")
ax2l.set_ylabel("acc_pa [m/s^2]")
ax2l.set_xlabel('Time [s]')
ax2r.set_xlabel('Time [s]')
plt.legend(loc="best")
fig.tight_layout()

# --------------------- Trajectory Reconstruction
regions_list = {k: pd.DataFrame([dtw.stride_list_[k].values.flatten()[[0, 15]],
                                 dtw.stride_list_[k].values.flatten()[[16, -1]]], columns=["start", "end"]).rename_axis(
    "gs_id") for k in ['Left Foot', 'Right Foot']}
trajectory_full_method = RtsKalman()
trajectory = StrideLevelTrajectory()
trajectory_full = RegionLevelTrajectory\
        (
        trajectory_method=trajectory_full_method, ori_method=None, pos_method=None
    )
trajectory = trajectory.estimate(
    data=dst,
    stride_event_list=ed.min_vel_event_list_,
    sampling_rate_hz=sampling_rate_hz
)
trajectory_full.estimate(
    data=dst,
    regions_of_interest=regions_list,
    sampling_rate_hz=sampling_rate_hz)
fig = plt.figure()
axl = fig.add_subplot(121, projection='3d')
axr = fig.add_subplot(122, projection='3d')

for i, ax in enumerate([axl, axr]): #foot_laterality):
    data_go = trajectory_full.position_[foot_laterality[i]].loc[0].values.T
    data_back = trajectory_full.position_[foot_laterality[i]].loc[1].values.T
    strides = dtw.stride_list_[foot_laterality[i]]
    strides_go = strides[strides.start<regions_list[foot_laterality[i]].end[0]]
    strides_back = strides[strides.start>regions_list[foot_laterality[i]].start[1]].reset_index(drop=True)
    strides_go -= strides_go.start[0]
    strides_back -= strides_back.start[0]

    colors = iter(cm.rainbow(np.linspace(0, 1, len(strides_go)+len(strides_back))))
    for istride in strides_go.iterrows():
        ax.scatter(data_go[0][istride[1].start: istride[1].end],
                   data_go[1][istride[1].start: istride[1].end],
                   data_go[2][istride[1].start: istride[1].end], color=next(colors))
    for istride in strides_back.iterrows():
        ax.scatter(data_back[0][istride[1].start: istride[1].end],
                   data_back[1][istride[1].start: istride[1].end],
                   data_back[2][istride[1].start: istride[1].end], color=next(colors))
# #         plt.scatter(x, y, color=c)
# # for istride in ed.min_vel_event_list_:
# #     test
# left_foot_0 = trajectory_full.position_['Left Foot'].loc[0].values.T
# right_foot_0 = trajectory_full.position_['Right Foot'].loc[0].values.T
# left_foot_1 = trajectory_full.position_['Left Foot'].loc[1].values.T
# right_foot_1 = trajectory_full.position_['Right Foot'].loc[1].values.T
# ax.scatter(left_foot_0[0], left_foot_0[1],zs=left_foot_0[2])
# ax.scatter(right_foot_0[0], right_foot_0[1],zs=right_foot_0[2])
#
# axr.scatter(left_foot_1[0], left_foot_1[1],zs=left_foot_1[2])
# axr.scatter(right_foot_1[0], right_foot_1[1],zs=right_foot_1[2])

# --------------------- Temporal Parameter Calculation
temporal_paras = TemporalParameterCalculation()
temporal_paras = temporal_paras.calculate(stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=sampling_rate_hz)

# --------------------- Spatial Parameter Calculation
spatial_paras = SpatialParameterCalculation()
spatial_paras = spatial_paras.calculate(
    stride_event_list=ed.min_vel_event_list_,
    positions=trajectory.position_,
    orientations=trajectory.orientation_,
    sampling_rate_hz=sampling_rate_hz,
)

# --------------------- Inspecting the Results
print(
    "The following number of strides were identified and parameterized for each sensor: {}".format(
        {k: len(v) for k, v in ed.min_vel_event_list_.items()}
    )
)
for k, v in temporal_paras.parameters_pretty_.items():
    v.plot()
    plt.title("All temporal parameters of sensor {}".format(k))

for k, v in spatial_paras.parameters_pretty_.items():
    v[["stride length [m]", "gait velocity [m/s]", "arc length [m]"]].plot()
    plt.title("All spatial parameters of sensor {}".format(k))

for k, v in spatial_paras.parameters_pretty_.items():
    v.filter(like="angle").plot()
    plt.title("All angle parameters of sensor {}".format(k))

