import numpy as np
import pandas as pd
from gaitmap.example_data import get_healthy_example_imu_data_not_rotated
from gaitmap.preprocessing import sensor_alignment
from gaitmap.stride_segmentation import BarthDtw, RoiStrideSegmentation
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.utils.rotations import flip_dataset, rotation_from_angle
from gaitmap.event_detection import RamppEventDetection
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory, RegionLevelTrajectory, RtsKalman
from gaitmap.parameters import TemporalParameterCalculation
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.gait_detection import UllrichGaitSequenceDetection
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import socket
import os
import itertools
from pathlib import Path
from Analysis_tools import *

np.random.seed(0)
foot_laterality = ["Left Foot", "Right Foot"]
first_time = True
repeat_strides = True

example_dataset = get_healthy_example_imu_data_not_rotated()
station = socket.gethostname()
if station == 'Laptop_naas':
    project_path = 'C:\\Users\\Nassila\\Documents\\Projects\\Multiple_sclerosis\\Dual_task\\'
elif station == 'w10lloppici':
    project_path = 'D:\\Multiple_Sclerosis\\Gait_dual_task\\'
    #Raw_data\\test_data_H\\sensors\\20210430-083735_Walk.csv'
else:
    raise ValueError('No idea where the raw data is')

# files = [ifile for ifile in (Path(project_path) / 'Raw_data/test_data_H/sensors').glob("*.csv")]
files = [ifile for ifile in (Path(project_path) / 'Raw_data//Sensor_data').glob("*.csv") if
         # 'Enan_S1_'
         any(keyw in ifile.stem for keyw in
             [ 'P2', #'P2059', 'P2074',
              # 'P2081', 'P2083', #'P1675', 'P1679', 'P1682', 'P1686', 'P1687', 'P1693', 'P1694', 'P1695'
         ]) and 'S1_' in ifile.stem
         ]
    # D80*6,5 (5 \  D190*6 \  D32*5,5 \ D22*5   \  D70*4,5 \ D266*4 \
    # D208*3,5 \ D230*3 \ D268*2,5 \ D396*2 \ D566*1,5 \ D104*1
for i, path_data in enumerate(files):
    if i<0:#path_data.stem not in [ #i<62:
        # Problematic, to debug later [

        # 'P128_E3_S1_NG_20220613-085946_Walk']:

        # P158_E6_S1_DT_20220621-132815_Walk
        # P1705_E6_S1_DT_20220624-101150_Walk
        # P1734_E6_S1_DT_20211122-133618_Walk
        # P204_E6_S1_DT_20210623-144818_Walk
        # P245_E6_S1_DT_20211210-105311_Walk
        # P245_E6_S1_NG_20211210-105030_Walk
        # P251_E6_S1_NG_20210906-114355_Walk
        # P408_E6_S1_DT_20220221-094552_Walk
        # P408_E6_S1_NG_20220221-094345_Walk
        # P537_E6_S1_DT_20210923-094632_Walk
        # P537_E6_S1_NG_20210923-094500_Walk
        # P58_E6_S1_DT_20221102-095041_Walk
        # P58_E6_S1_NG_20221102-094754_Walk
        # P63_E6_S1_DT_20210611-111352_Walk
        # P63_E6_S1_NG_20210611-111013_Walk
        # P816_E6_S1_DT_20210614-102915_Walk


        # P1045_E6,5_S1_DT_20210701-150016_Walk
        # P1192_E6,5_S1_DT_20220426-093452_Walk
        # P1192_E6,5_S1_NG_20220426-093204_Walk
        # P129_E6,5_S1_DT_20210608-112715_Walk
        # P129_E6,5_S1_NG_20210608-112415_Walk
        # P12_E6,5_S1_DT_20220315-111515_Walk
        # P1366_E6,5_S1_DT_20220104-125343_Walk
        # P1366_E6,5_S1_NG_20220104-124958_Walk
        # P1588_E6,5_S1_DT_20220622-142244_Walk
        # P1588_E6,5_S1_NG_20220622-142051_Walk
        # P1616_E6,5_S1_DT_20211007-105220_Walk
        # P1816_E6,5_S1_DT_20220909-120542_Walk
        # P1816_E6,5_S1_NG_20220909-120102_Walk
        # P199_E6,5_S1_DT_20230228-093313_Walk
        # P199_E6,5_S1_NG_20230228-092851_Walk
        # P218_E6,5_S1_DT_20220309-103513_Walk
        # P218_E6,5_S1_NG_20220309-103220_Walk
        # P221_E6,5_S1_DT_20210927-152148_Walk
        # P221_E6,5_S1_NG_20210927-151838_Walk
        # P322_E6,5_S1_DT_20211123-102343_Walk
        # P322_E6,5_S1_NG_20211123-102201_Walk

        # ]
        continue
    if 'Enan_S' in path_data.stem:
        print(path_data.stem)
        continue

    trial_name = path_data.stem
    print('**********************************************************************')
    print(f'Processing participant : {i}/{len(files)}')
    print(trial_name)
    data_set = pd.read_csv(path_data, sep='\t')
    data_set.iloc[:, 1:] = data_set.iloc[:, 1:].rolling(10, min_periods=1).mean()
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

    """************************************************************
    ------------------------- Identify ROI ------------------------
    """
    # Convert data to foot-frame
    bf_data = convert_to_fbf(dataset_sf_al_to_gr, left_like="Left", right_like="Right")
    previous_gait_regions = os.path.isfile(str(path_data.parent)+f'\\ROI\ROI_{trial_name}.npy')
    previous_gait_strides = os.path.isfile(str(path_data.parent)+f'\\Stride_detection\\StrideBound_{trial_name}.npy')

    if previous_gait_regions:
        first_time = False
        while True:
            repeat = input(
                f"The ROI identification has been done for {trial_name}, do you want to repeat it? (No [n],"
                f" repeat all [y], repeat only strides ['s'], repeat only from trajectory ['t'])")
            if repeat.lower() not in ('y', 'n', 's', 't'):
                print("Not an appropriate choice. Please input y, n, s or t:")
            else:
                break

        if repeat == 'n':
            repeat = False
            if not previous_gait_strides:
                repeat_strides = True
                repeat_traj = True
                print('Could not find stride folder, mandatory repetition')
            else:
                repeat_strides = False
                continue
        elif repeat == 'y':
            repeat = True
            repeat_strides = True
            repeat_traj = True
            print('Repeating gait region detection correction')
        elif repeat == 's':
            repeat_strides = True
            repeat_traj = True
            repeat = False
            print('Repeating strides detection correction')
        elif repeat == 't':
            repeat_traj = True
            repeat_strides = False
            repeat = False
        else:
            raise ValueError
    else:
        first_time=True

    if first_time or (not first_time and repeat):
        gsd = UllrichGaitSequenceDetection(window_size_s=4)
        gsd = gsd.detect(data=bf_data, sampling_rate_hz=sampling_rate_hz)
        gsd = plot_gait_sequence(gsd, foot_laterality, title= str(path_data.parent)+f'\\ROI\ROI_{trial_name}.svg', trial_name = f"{i}_{trial_name}")
        regions_list = gsd.gait_sequences_
        np.save(str(path_data.parent) + f'\\ROI\ROI_{trial_name}.npy', regions_list)
    elif repeat_strides or repeat_traj:
        regions_list = np.load(str(path_data.parent)+f'\\ROI\ROI_{trial_name}.npy', allow_pickle=True).item()
    else:
        continue

    
    if not regions_list[foot_laterality[0]].shape[0] == regions_list[foot_laterality[1]].shape[0] == 2:
        raise ValueError('Found more\less than 2 ROI for one of the feet')
    else:

        """************************************************************
        --------------------- Stride Segmentation ---------------------
        """
        you_are_not_happy = True
        while you_are_not_happy:
            dtw = BarthDtw(max_cost=4.3, min_match_length_s=0.01, max_match_length_s=5)

            roi_seg = RoiStrideSegmentation(segmentation_algorithm=dtw,)
            roi_seg = roi_seg.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz, regions_of_interest=regions_list)
            # dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

            # Check detected strides
            plot_strides(roi_seg, time, sampling_rate_hz, foot_laterality,
                     save_path=str(path_data.parent)+f'/Stride_detection/Stride_detection_{trial_name}.svg')
            you_are_not_happy = False #input(f"Are you happy? (No [0], Yes[1])").lower().strip() != '1'

        stride_list = roi_seg.stride_list_
        # Drop first and last strides
        for sensor in foot_laterality:
            stride_list[sensor] =  pd.concat([
                stride_list[sensor][stride_list[sensor].gs_id == 0].iloc[1:-1, :],
                stride_list[sensor][stride_list[sensor].gs_id == 1].iloc[1:-1, :]
            ], ignore_index=True)


        # if np.abs(stride_list['Left Foot'].shape[0]-stride_list['Right Foot'].shape[0]) > 3:
        #     print(f'Probably a problem in cycle detection for one of the feet for {trial_name}')
        #     continue
        # else:
        dtw_path = str(path_data.parent)+f'/Stride_detection/StrideBound_{trial_name}.npy'
        stride_dict = stride_list
        stride_dict.update({'Rate': sampling_rate_hz})
        np.save(dtw_path, stride_dict)

        """************************************************************
        --------------------- Event detection ---------------------
        """
        stride_list_sid = stride_list.copy()
        stride_list_sid.popitem()
        for sensor in foot_laterality:
            stride_list_sid[sensor] = stride_list_sid[sensor].assign(s_id=np.arange(0, stride_list_sid[sensor].shape[0]))
        ed = RamppEventDetection()
        ed = ed.detect(data=bf_data,
                       stride_list=stride_list_sid,
                       sampling_rate_hz=sampling_rate_hz,
                       )

        # Check detected events
        plot_events(ed, bf_data, time, sampling_rate_hz, foot_laterality,
                    save_path=str(path_data.parent)+f'\\Events\Event_detection_{trial_name}.svg')

        """************************************************************
        ------------------ Trajectory reconstruction ------------------
        """

        # regions_list = gsd.gait_sequences_
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
        plot_trajectory(trajectory_full, stride_list_sid, foot_laterality,
                        save_path=str(path_data.parent)+f'\\Trajectory\Trajectory_{trial_name}.svg')
        # fig = plt.figure()
        # axl = fig.add_subplot(121, projection='3d')
        # axr = fig.add_subplot(122, projection='3d')
        #
        # for i, ax in enumerate([axl, axr]): #foot_laterality):
        #     data_go = trajectory_full.position_[foot_laterality[i]].loc[0].values.T
        #     data_back = trajectory_full.position_[foot_laterality[i]].loc[1].values.T
        #     strides = dtw.stride_list_[foot_laterality[i]]
        #     strides_go = strides[strides.start<regions_list[foot_laterality[i]].end[0]]
        #     strides_back = strides[strides.start>regions_list[foot_laterality[i]].start[1]].reset_index(drop=True)
        #     strides_go -= strides_go.start[0]
        #     strides_back -= strides_back.start[0]
        #
        #     colors = iter(cm.rainbow(np.linspace(0, 1, len(strides_go)+len(strides_back))))
        #     for istride in strides_go.iterrows():
        #         ax.scatter(data_go[0][istride[1].start: istride[1].end],
        #                    data_go[1][istride[1].start: istride[1].end],
        #                    data_go[2][istride[1].start: istride[1].end], color=next(colors))
        #     for istride in strides_back.iterrows():
        #         ax.scatter(data_back[0][istride[1].start: istride[1].end],
        #                    data_back[1][istride[1].start: istride[1].end],
        #                    data_back[2][istride[1].start: istride[1].end], color=next(colors))
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
        temporal_path = str(path_data.parent)+f'\\Temporal_data\Temporal_{trial_name}.npy'
        np.save(temporal_path, temporal_paras.parameters_pretty_)


        # --------------------- Spatial Parameter Calculation
        spatial_paras = SpatialParameterCalculation()
        spatial_paras = spatial_paras.calculate(
            stride_event_list=ed.min_vel_event_list_,
            positions=trajectory.position_,
            orientations=trajectory.orientation_,
            sampling_rate_hz=sampling_rate_hz,
        )
        spatial_path = str(path_data.parent)+f'\\Spatial_data\Spatial_{trial_name}.npy'
        np.save(spatial_path, spatial_paras.parameters_pretty_)

        # --------------------- Inspecting the Results
        print(
            "The following number of strides were identified and parameterized for each sensor: {}".format(
                {k: len(v) for k, v in ed.min_vel_event_list_.items()}
            )
        )
        # for k, v in temporal_paras.parameters_pretty_.items():
        #     v.plot()
        #     plt.title("All temporal parameters of sensor {}".format(k))
        #
        # for k, v in spatial_paras.parameters_pretty_.items():
        #     v[["stride length [m]", "gait velocity [m/s]", "arc length [m]"]].plot()
        #     plt.title("All spatial parameters of sensor {}".format(k))
        #
        # for k, v in spatial_paras.parameters_pretty_.items():
        #     v.filter(like="angle").plot()
        #     plt.title("All angle parameters of sensor {}".format(k))

        # print('yep')