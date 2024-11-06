import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import CheckButtons, Button, SpanSelector
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Polygon
def on_pick(fig, event):
    event.artist.set_visible(not event.artist.get_visible())
    fig.canvas.draw()

def onpick(event, events_to_remove, f, check_box, check_lat, t_remove):
    if isinstance(event.artist, Line2D) and check_box.get_status() == [0,0,1]:
        thisline = event.artist
        thisline.set_color('k')
        xdata = thisline.get_xdata()[0]
        events_to_remove.append([np.argwhere(check_lat.get_status())[0][0],xdata])
        print('onpick line:', xdata)
        t_remove.set_text(int(t_remove.get_text()) + 1)
    elif isinstance(event.artist, Polygon) and check_box.get_status() == [0,0,1]:
        thisspan = event.artist
        thisspan.remove()
    elif isinstance(event.artist, Text):
        f.canvas.stop_event_loop()
        event.artist.set_backgroundcolor('green')

def onpick_span(event, events_to_add, events_to_remove, f, check_box, check_lat,
                t_start, t_rest, t_remove):
    if isinstance(event.artist, Polygon) and check_box.get_status() == [0, 0, 1]:
        thisspan = event.artist
        events_to_remove.append([np.argwhere(check_lat.get_status())[0][0],
                                 list(set(thisspan.xy[:, 0]))])
        t_remove.set_text(int(t_remove.get_text()) + 1)
        thisspan.set_color('r')
        # thisspan.remove()
    elif isinstance(event.artist, Polygon) and check_box.get_status() == [1, 0 , 0]:
        check_box.eventson = False
        thisspan = event.artist
        events_to_add.append([np.argwhere(check_lat.get_status())[0][0],
                                 np.sort(list(set(thisspan.xy[:, 0])))])
        events_to_remove.append([np.argwhere(check_lat.get_status())[0][0],
                                 list(set(thisspan.xy[:, 0]))])
        t_remove.set_text(int(t_remove.get_text()) + 1)
        thisspan.set_color('m')
        check_box.set_active(0)
        check_box.set_active(1)
        check_box.eventson = True
        t_start.set_text(int(t_start.get_text()) + 1)
    elif isinstance(event.artist, Polygon) and check_box.get_status() == [0, 1, 0]:
        check_box.eventson = False
        thisspan = event.artist
        events_to_add.append([np.argwhere(check_lat.get_status())[0][0],
                              np.sort(list(set(thisspan.xy[:, 0])))])
        print(f'new span between [ {events_to_add[-2][1][0]}, {events_to_add[-1][1][0]} ]')
        newspan = thisspan.axes.axvspan(events_to_add[-2][1][0],events_to_add[-1][1][0],
                                        alpha=0.3, color="c")
        newspan.set_color('c')
        check_box.set_active(0)
        check_box.set_active(1)
        check_box.eventson = True
        t_rest.set_text(int(t_rest.get_text()) + 1)

    elif isinstance(event.artist, Text):
        f.canvas.stop_event_loop()
        event.artist.set_backgroundcolor('green')
def double_left_mouse(event, events_to_add, f, check_box, check_lat, t_start, t_rest):
    if event.button == plt.MouseButton.LEFT and event.dblclick == True and check_box.get_status()==[1,0,0]:
        check_box.eventson = False
        events_to_add.append([np.argwhere(check_lat.get_status())[0][0],event.xdata])
        line = f.axes[np.argwhere(check_lat.get_status())[0][0]].axvline(event.xdata, color='g', linestyle="--",  label='Manually identified start')
        f.axes[np.argwhere(check_lat.get_status())[0][0]].draw_artist(line)
        check_box.set_active(0)
        check_box.set_active(1)
        check_box.eventson = True
        t_start.set_text(int(t_start.get_text())+1)
    elif event.button == plt.MouseButton.LEFT and event.dblclick == True and check_box.get_status()==[0,1,0]:
        events_to_add.append([np.argwhere(check_lat.get_status())[0][0], event.xdata])
        line = f.axes[np.argwhere(check_lat.get_status())[0][0]].axvline(event.xdata, color='r', linestyle="--", label='Manually identified end')
        f.axes[np.argwhere(check_lat.get_status())[0][0]].draw_artist(line)
        t_rest.set_text(int(t_rest.get_text()) + 1)
        check_box.eventson = False
        check_box.set_active(0)
        check_box.set_active(1)
        check_box.eventson = True

def plot_new_range(event, ax, gsd, events_to_add, events_to_remove, ):
    laterality = {0: "Left Foot", 1: "Right Foot"}
    if len(events_to_remove) > 0:
        for lat_index, iremove in events_to_remove:
            row_to_drop_index = gsd.gait_sequences_[laterality[lat_index]][
                                                                gsd.gait_sequences_[laterality[
                                                                    lat_index]].start == iremove * gsd.sampling_rate_hz].index
            ax[lat_index].axvline(iremove, color='w')
            ax[lat_index].axvline(
                gsd.gait_sequences_[laterality[lat_index]].loc[row_to_drop_index].end.values / gsd.sampling_rate_hz,
                color='w')

            gsd.gait_sequences_[laterality[lat_index]].drop(row_to_drop_index,
                                                            inplace=True)

    gsd.gait_sequences_['Left Foot'].reset_index(inplace=True, drop=True)
    gsd.gait_sequences_['Right Foot'].reset_index(inplace=True, drop=True)

    if len(events_to_add)>0:
        for i, (lat_index, iadd) in enumerate(events_to_add[::2]):
            next_row = [gsd.gait_sequences_[laterality[0]].shape[0],
                        gsd.gait_sequences_[laterality[1]].shape[0]]
            ax[lat_index].axvspan(iadd, events_to_add[2 * i + 1][1], facecolor="grey", alpha=0.6, zorder=10)
            gsd.gait_sequences_[laterality[lat_index]].loc[next_row[lat_index]]=[next_row[lat_index],
                                                                         int(iadd*gsd.sampling_rate_hz),
                                                                         int(events_to_add[2*i+1][1] * gsd.sampling_rate_hz)
                                                                         ]


    for ifoot in ['Left Foot', 'Right Foot']:
        gsd.gait_sequences_[ifoot].sort_values('start', inplace =True)

        gsd.gait_sequences_[ifoot].reset_index(inplace=True, drop=True)

        gsd.gait_sequences_[ifoot].gs_id = gsd.gait_sequences_[ifoot].index.values

def update_strides_list(event, ax, roi_seg, strides_to_add, strides_to_remove, axtext):
    laterality = {0: "Left Foot", 1: "Right Foot"}
    if len(strides_to_remove) > 0:
        for lat_index, iremove in strides_to_remove:
            iremove = np.sort(iremove)
            iremove_indices = np.multiply(iremove, roi_seg.sampling_rate_hz)

            row_to_drop_index = roi_seg.stride_list_[laterality[lat_index]][
                roi_seg.stride_list_[laterality[lat_index]].start == iremove_indices[0]].index
            roi_seg.stride_list_[laterality[lat_index]].drop(row_to_drop_index, inplace=True)

    roi_seg.stride_list_['Left Foot'].reset_index(inplace=True, drop=True)
    roi_seg.stride_list_['Right Foot'].reset_index(inplace=True, drop=True)

    if len(strides_to_add) > 0:
        for i, (lat_index, iadd) in enumerate(strides_to_add[::2]):
            next_row = [roi_seg.stride_list_[laterality[0]].shape[0],
                        roi_seg.stride_list_[laterality[1]].shape[0]]
            gs_id_shift = [roi_seg.stride_list_[laterality[0]][roi_seg.stride_list_[laterality[0]].gs_id == 0].values.max(),
                           roi_seg.stride_list_[laterality[1]][roi_seg.stride_list_[laterality[1]].gs_id == 0].values.max()]
            index_iadd = np.multiply(roi_seg.sampling_rate_hz, iadd).astype(int)
            current_id = [0 if index_iadd[1]<gs_id_shift[lat_index]  else 1][0]
            new_row = [current_id, index_iadd[0], int(strides_to_add[2*i+1][1][0]*roi_seg.sampling_rate_hz)]
            roi_seg.stride_list_[laterality[lat_index]].loc[next_row[lat_index]]=new_row


    for ifoot in ['Left Foot', 'Right Foot']:
        roi_seg.stride_list_[ifoot].sort_values('start', inplace=True)
        roi_seg.stride_list_[ifoot].reset_index(inplace=True, drop=True)
        roi_seg.stride_list_[ifoot].index.names = ['s_id']

    axtext.set_color('k')
    axtext.set_text(f'Modification status -> Updated')
    print('Stride list updated')

def on_keyboard_stride(event, fig, text_artist, stride, text, action_box, lat_box):
    print('press', event.key)
    if event.key == 'a':
        if (len(stride)>0 and text.get_text()== 'Modification status -> Updated')\
                or len(stride) == 0:
            fig.canvas.stop_event_loop()
            text_artist.set_backgroundcolor('green')
        else:
            text.set_color('r')
            text.set_text('Modification status -> Please update before closing')

    elif event.key == 't':
        if not lat_box.get_status()[1]:
            lat_box.set_active(1)
        if lat_box.get_status()[0]:
            lat_box.set_active(0)

    elif event.key == 'e':
        if not lat_box.get_status()[0]:
            lat_box.set_active(0)
        if lat_box.get_status()[1]:
            lat_box.set_active(1)

    elif event.key == '1':
        if not action_box.get_status()[0]:
            action_box.set_active(0)
        if action_box.get_status()[1]:
            action_box.set_active(1)
        if action_box.get_status()[2]:
            action_box.set_active(2)

    elif event.key == '2':
        if not action_box.get_status()[1]:
            action_box.set_active(1)
        if action_box.get_status()[0]:
            action_box.set_active(0)
        if action_box.get_status()[2]:
            action_box.set_active(2)

    elif event.key == '3':
        if not action_box.get_status()[2]:
            action_box.set_active(2)
        if action_box.get_status()[1]:
            action_box.set_active(1)
        if action_box.get_status()[0]:
            action_box.set_active(0)

def on_keyboard(event, fig, text_artist, action_box, lat_box):
    print('press', event.key)
    if event.key == 'a':
        fig.canvas.stop_event_loop()
        text_artist.set_backgroundcolor('green')
    elif event.key == 't':
        if not lat_box.get_status()[1]:
            lat_box.set_active(1)
        if lat_box.get_status()[0]:
            lat_box.set_active(0)

    elif event.key == 'e':
        if not lat_box.get_status()[0]:
            lat_box.set_active(0)
        if lat_box.get_status()[1]:
            lat_box.set_active(1)

    elif event.key == '1':
        if not action_box.get_status()[0]:
            action_box.set_active(0)
        if action_box.get_status()[1]:
            action_box.set_active(1)
        if action_box.get_status()[2]:
            action_box.set_active(2)

    elif event.key == '2':
        if not action_box.get_status()[1]:
            action_box.set_active(1)
        if action_box.get_status()[0]:
            action_box.set_active(0)
        if action_box.get_status()[2]:
            action_box.set_active(2)

    elif event.key == '3':
        if not action_box.get_status()[2]:
            action_box.set_active(2)
        if action_box.get_status()[1]:
            action_box.set_active(1)
        if action_box.get_status()[0]:
            action_box.set_active(0)


def plot_gait_sequence(gsd, foot_laterality, title, check_plot_all = False, trial_name=''):
    gait_sequences = gsd.gait_sequences_
    roi_to_remove = []
    roi_to_add = []
    fig, ax1 = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(20, 10))
    ax1[0].set_ylabel("gyro_ml [deg]")
    axButn1 = plt.axes([0.1, 0.1, 0.1, 0.1])
    btn1 = Button(axButn1, 'Show Final ROI', color='grey')
    final_plot = lambda x: plot_new_range(x, ax1, gsd, roi_to_add, roi_to_remove)

    btn1.on_clicked(final_plot)
    for i, sensor in enumerate(foot_laterality):
        ax1[i].plot(gsd.data.index.values, gsd.data[sensor]['gyr_ml'], label="gyr_ml")
        ax1[i].set_xlabel("Time [s]")
        ax1[i].set_title(sensor)
        if len(gsd.start_[sensor]) != 0:
            start = gsd.data.index.values[gsd.start_[sensor]]
            if gsd.data.index.values.shape[0] in gsd.end_[sensor]:
                end = gsd.data.index.values[gsd.end_[sensor]-1]
            else:
                end = gsd.data.index.values[gsd.end_[sensor]]
            for istart, iend in zip(start, end):
                tv = ax1[i].axvline(istart, color='g', label='start')
                tv.set_picker(True)
                ax1[i].axvline(iend, color='r', label='end')
                ts = ax1[i].axvspan(istart, iend, facecolor="grey", alpha=0.6)
                ts.set_picker(True)

    plt.suptitle(trial_name)
    plt.subplots_adjust(left=0.3, right=0.95, bottom=0.2)
    axcheck = plt.axes([0.03, 0.3, 0.2, 0.15])
    add_remove_check_box = CheckButtons(axcheck,
                                        labels=['Add start',
                                                'Add rest\n(immediately after add start)',
                                                'Select starts to remove cycle'],
                                        actives=[0, 0, 1])
    check_color = ['g', 'r', 'k']
    [ilabel.set_color(check_color[icolor]) for icolor, ilabel in enumerate(add_remove_check_box.labels)]
    axcheck.text(0.03, -0.10, "Modification status", ha='left', va='center', transform=axcheck.transAxes, color='k')

    axcheck_lat = plt.axes([0.03, 0.6, 0.2, 0.15])
    detect_lat_check_box = CheckButtons(axcheck_lat,
                                        labels=[
                                            'Left',
                                            'Right',
                                        ],
                                        actives=[1,0])

    tax = ax1[0].text(0.7, -0.2, '✓ Correction done',
                ha='left', va='center', transform=ax1[0].transAxes,
                fontsize=20, picker=10, bbox=dict(edgecolor='black', alpha=0.6, linewidth=0))

    t_add_start = axcheck.text(0.03, -0.2, 0, ha='left', va='center', transform=axcheck.transAxes, color='g')
    t_add_rest = axcheck.text(0.03, -0.3, 0, ha='left', va='center', transform=axcheck.transAxes, color='r')
    t_remove = axcheck.text(0.03, -0.4, 0, ha='left', va='center', transform=axcheck.transAxes, color='k')

    # if (check_plot_all or
    #         (not gsd.start_[foot_laterality[0]].shape == gsd.start_[foot_laterality[1]].shape == (2,))):
    fig.canvas.mpl_connect('pick_event',
                       lambda event: onpick(event, roi_to_remove,
                                            fig, add_remove_check_box, detect_lat_check_box,
                                            t_remove))
    fig.canvas.mpl_connect('button_press_event',
                     lambda event: double_left_mouse(event, roi_to_add,
                                                     fig, add_remove_check_box, detect_lat_check_box,
                                                     t_add_start, t_add_rest)
                       )
    fig.canvas.mpl_connect('key_press_event', lambda event: on_keyboard(event, fig, tax, add_remove_check_box, detect_lat_check_box))

    fig.canvas.start_event_loop(timeout=-1)

    plt.savefig(title)
    plt.close()
    return gsd

def plot_strides(roi_seg, time, sampling_rate_hz, foot_laterality, save_path):
    strides_to_remove = []
    strides_to_add = []
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(20, 10))

    for i, sensor in enumerate(foot_laterality):
        cost_funcs = {roi: dtw.cost_function_[sensor] for roi, dtw in roi_seg.instances_per_roi_[sensor].items()}
        cost_mat = {roi: dtw.acc_cost_mat_[sensor] for roi, dtw in roi_seg.instances_per_roi_[sensor].items()}
        full_cost_matrix = np.full((len(roi_seg.segmentation_algorithm.template.get_data()), len(roi_seg.data)),
                                   np.nan)

        axs[0, i].plot(time, roi_seg.data[sensor]["gyr_ml"].reset_index(drop=True))
        axs[0, i].set_ylabel("gyro [deg/s]")
        for roi, (start, end) in roi_seg.regions_of_interest[sensor][["start", "end"]].iterrows():
            axs[2, i].imshow(cost_mat[roi], aspect="auto",
                             extent=[start / sampling_rate_hz, end / sampling_rate_hz, 120, 0])
            axs[1, i].plot(np.arange(start, end)/sampling_rate_hz, cost_funcs[roi], c="C0")

        axs[1, i].set_ylabel("dtw cost [a.u.]")
        axs[1, i].axhline(roi_seg.segmentation_algorithm.max_cost, color="k", linestyle="--")
        axs[2, i].set_ylabel("template position [#]")
        for roi_id, dtw_instance in roi_seg.instances_per_roi_[sensor].items():
            roi_start = roi_seg.regions_of_interest[sensor].loc[roi_id]["start"]
            for p in dtw_instance.paths_[sensor]:
                axs[2, i].plot((p.T[1] + roi_start)/sampling_rate_hz, p.T[0])
            for start, end in dtw_instance.matches_start_end_original_[sensor]:
                axs[1, i].axvspan((start + roi_start) / sampling_rate_hz,
                                      (end + roi_start) / sampling_rate_hz, alpha=0.3, color="g")

        drop_index = len(roi_seg.instances_per_roi_[sensor][0].matches_start_end_original_[sensor])
        for ispan,  s in roi_seg.stride_list_[sensor][["start", "end"]].iterrows():
            if ispan in [0, drop_index-1, drop_index, roi_seg.stride_list_[sensor].shape[0]-1]:
                td = axs[0, i].axvspan(*s / sampling_rate_hz, alpha=0.3, color="r")
                td.set_picker(True)
            else:
                ts= axs[0, i].axvspan(*s / sampling_rate_hz, alpha=0.3, color="g")
                ts.set_picker(True)


        axs[2, i].set_xlabel("time [s]")
        axs[0, i].set_title(sensor)

    plt.subplots_adjust(left=0.3, right=0.95, bottom=0.2)
    axcheck = plt.axes([0.03, 0.3, 0.2, 0.15])
    add_remove_check_box = CheckButtons(axcheck,
                                        labels=[
                                                'Add spans (first)',
                                                'Add spans (second)',
                                                'Select starts to remove cycle'],
                                        actives=[0, 0, 0])
    check_color = ['m','c','r']
    [ilabel.set_color(check_color[icolor]) for icolor, ilabel in enumerate(add_remove_check_box.labels)]
    axt = axcheck.text(0.03, -0.10, "Modification status", ha='left', va='center',
                       transform=axcheck.transAxes, color='k')

    t_add_start = axcheck.text(0.03, -0.2, 0, ha='left', va='center', transform=axcheck.transAxes, color='m')
    t_add_rest = axcheck.text(0.03, -0.3, 0, ha='left', va='center', transform=axcheck.transAxes, color='c')
    t_remove = axcheck.text(0.03, -0.4, 0, ha='left', va='center', transform=axcheck.transAxes, color='r')
    axcheck_lat = plt.axes([0.03, 0.6, 0.2, 0.15])
    detect_lat_check_box = CheckButtons(axcheck_lat,
                                        labels=[
                                            'Left',
                                            'Right',
                                        ],
                                        actives=[1, 0])

    tax = axs[2,0].text(0.7, -0.2, '✓ Correction done',
                ha='left', va='center', transform=axs[2,0].transAxes,
                fontsize=20, picker=10, bbox=dict(edgecolor='black', alpha=0.6, linewidth=0))

    axButn1 = plt.axes([0.1, 0.1, 0.1, 0.1])
    btn1 = Button(axButn1, 'Update stride list', color='grey')
    final_plot = lambda x: update_strides_list(x, axs, roi_seg, strides_to_add, strides_to_remove, axt)

    btn1.on_clicked(final_plot)
    # fig.tight_layout()

    fig.canvas.mpl_connect('pick_event',
                           lambda event: onpick_span(event, strides_to_add, strides_to_remove,
                                                fig, add_remove_check_box, detect_lat_check_box,
                                                t_add_start, t_add_rest, t_remove))

    fig.canvas.mpl_connect('key_press_event', lambda event: on_keyboard_stride(event, fig, tax, strides_to_remove, axt, add_remove_check_box, detect_lat_check_box))

    fig.canvas.start_event_loop(timeout=-1)
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def plot_events(ed, bf_data, time, sampling_rate_hz, foot_laterality, save_path):
    fig, [[ax1l, ax1r], [ax2l, ax2r]] = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))
    ax1l.plot(time, bf_data.reset_index(drop=True)["Left Foot"][["gyr_ml"]])
    ax2l.plot(time, bf_data.reset_index(drop=True)["Left Foot"][["acc_pa"]])
    ax1r.plot(time, bf_data.reset_index(drop=True)["Right Foot"][["gyr_ml"]])
    ax2r.plot(time, bf_data.reset_index(drop=True)["Right Foot"][["acc_pa"]])

    ic_idx = [ed.min_vel_event_list_[k]["ic"].to_numpy().astype(int) for k in ["Left Foot", "Right Foot"]]
    tc_idx = [ed.min_vel_event_list_[k]["tc"].to_numpy().astype(int) for k in ["Left Foot", "Right Foot"]]
    min_vel_idx = [ed.min_vel_event_list_[k]["min_vel"].to_numpy().astype(int) for k in ["Left Foot", "Right Foot"]]

    for iside, side in enumerate([[ax1l, ax2l], [ax1r, ax2r]]):
        for ax, sensor in zip(side, ["gyr_ml", "acc_pa"]):
            for i, stride in ed.min_vel_event_list_[foot_laterality[iside]].iterrows():
                ax.axvline(stride["start"] / sampling_rate_hz, color="g")
                ax.axvline(stride["end"] / sampling_rate_hz, color="r")

            ax.scatter(ic_idx[iside] / sampling_rate_hz,
                       bf_data[foot_laterality[iside]][sensor].to_numpy()[ic_idx[iside]], marker="*", s=100,
                       color="r", zorder=3, label="ic",
                       )

            ax.scatter(tc_idx[iside] / sampling_rate_hz,
                       bf_data[foot_laterality[iside]][sensor].to_numpy()[tc_idx[iside]], marker="p", s=50,
                       color="g", zorder=3, label="tc",
                       )

            ax.scatter(min_vel_idx[iside] / sampling_rate_hz,
                       bf_data[foot_laterality[iside]][sensor].to_numpy()[min_vel_idx[iside]],
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
    plt.savefig(save_path)
    plt.close()

def plot_trajectory(trajectory_full, dtw, foot_laterality, save_path):
    fig = plt.figure()
    axl = fig.add_subplot(121, projection='3d')
    axr = fig.add_subplot(122, projection='3d')
    for i, ax in enumerate([axl, axr]):  # foot_laterality):
        data_go = trajectory_full.position_[foot_laterality[i]].loc[0].values.T
        data_back = trajectory_full.position_[foot_laterality[i]].loc[1].values.T
        strides = dtw[foot_laterality[i]]
        strides_go = strides[strides.gs_id==0]
        strides_back = strides[strides.gs_id==1].reset_index(drop=True)
        strides_go -= strides_go.start[0]
        strides_back -= strides_back.start[0]

        colors = iter(cm.rainbow(np.linspace(0, 1, len(strides_go) + len(strides_back))))
        for istride in strides_go.iterrows():
            ax.scatter(data_go[0][istride[1].start: istride[1].end],
                       data_go[1][istride[1].start: istride[1].end],
                       data_go[2][istride[1].start: istride[1].end], color=next(colors))
        for istride in strides_back.iterrows():
            ax.scatter(data_back[0][istride[1].start: istride[1].end],
                       data_back[1][istride[1].start: istride[1].end],
                       data_back[2][istride[1].start: istride[1].end], color=next(colors))
    axl.set_title('Left Foot')
    axr.set_title('Right Foot')
    plt.savefig(save_path)
    plt.close()