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

np.random.seed(0)

example_dataset = get_healthy_example_imu_data_not_rotated()
data_set = pd.read_csv('D:\\Multiple_Sclerosis\\Gait_dual_task\\Raw_data\\test_data_H\\sensors\\20210430-083735_Walk'
                       '.csv', sep='\t')
sampling_rate_hz = 1/data_set.time[1] #204.8
data_set_foot = data_set[[icol for icol in data_set.columns if 'Foot' in icol or icol=='time']]

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
dtw = BarthDtw()
# Convert data to foot-frame
bf_data = convert_to_fbf(dataset_sf_al_to_gr, left_like="Left", right_like="Right")
dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

# Check detected strides
sensor = "Left Foot"
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
dtw.data[sensor]["gyr_ml"].reset_index(drop=True).plot(ax=axs[0])
axs[0].set_ylabel("gyro [deg/s]")
axs[1].plot(dtw.cost_function_[sensor])
axs[1].set_ylabel("dtw cost [a.u.]")
axs[1].axhline(dtw.max_cost, color="k", linestyle="--")
axs[2].imshow(dtw.acc_cost_mat_[sensor], aspect="auto")
axs[2].set_ylabel("template position [#]")
for p in dtw.paths_[sensor]:
    axs[2].plot(p.T[1], p.T[0])
for s in dtw.matches_start_end_original_[sensor]:
    axs[1].axvspan(*s, alpha=0.3, color="g")
for _, s in dtw.stride_list_[sensor][["start", "end"]].iterrows():
    axs[0].axvspan(*s, alpha=0.3, color="g")

axs[0].set_xlabel("time [#]")
fig.tight_layout()

# --------------------- Event detection
ed = RamppEventDetection()
ed = ed.detect(data=bf_data, stride_list=dtw.stride_list_, sampling_rate_hz=sampling_rate_hz)

# Check detected events
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 5))
ax1.plot(bf_data.reset_index(drop=True)["Left Foot"][["gyr_ml"]])
ax2.plot(bf_data.reset_index(drop=True)["Left Foot"][["acc_pa"]])

ic_idx = ed.min_vel_event_list_["Left Foot"]["ic"].to_numpy().astype(int)
tc_idx = ed.min_vel_event_list_["Left Foot"]["tc"].to_numpy().astype(int)
min_vel_idx = ed.min_vel_event_list_["Left Foot"]["min_vel"].to_numpy().astype(int)

for ax, sensor in zip([ax1, ax2], ["gyr_ml", "acc_pa"]):
    for i, stride in ed.min_vel_event_list_["Left Foot"].iterrows():
        ax.axvline(stride["start"], color="g")
        ax.axvline(stride["end"], color="r")

    ax.scatter(ic_idx, bf_data["Left Foot"][sensor].to_numpy()[ic_idx],marker="*",s=100,
               color="r", zorder=3, label="ic",
    )

    ax.scatter( tc_idx, bf_data["Left Foot"][sensor].to_numpy()[tc_idx], marker="p", s=50,
                color="g", zorder=3, label="tc",
    )

    ax.scatter(min_vel_idx, bf_data["Left Foot"][sensor].to_numpy()[min_vel_idx], marker="s", s=50,
               color="y", zorder=3, label="min_vel",
    )
    ax.grid(True)

ax1.set_title("Events of min_vel strides")
ax1.set_ylabel("gyr_ml (°/s)")
ax2.set_ylabel("acc_pa [m/s^2]")
plt.legend(loc="best")
fig.tight_layout()

# --------------------- Trajectory Reconstruction
regions_list = {k: pd.DataFrame([dtw.stride_list_[k].values.flatten()[[0, -1]]], columns=["start", "end"]).rename_axis(
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
ax = fig.add_subplot(111, projection='3d')
left_foot = trajectory_full.position_['Left Foot'].values.T
right_foot = trajectory_full.position_['Right Foot'].values.T
ax.scatter(left_foot[0], left_foot[1],zs=left_foot[2])
ax.scatter(right_foot[0], right_foot[1],zs=right_foot[2])

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

