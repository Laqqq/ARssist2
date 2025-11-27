import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hl2ss", "viewer"))
)

import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_io
import hl2ss_mx
import hl2ss_utilities
import hl2ss_3dcv
import json
from collections import deque
import pickle
import time
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import concurrent.futures

from utils.utils import PVCalibrationToOpenCVFormat, memsize
from ultralytics import YOLO
from pose_estimation.pose_estimation import process_image
from pose_estimation.nn_shaft_detection import SAM2PointTracker
from pose_refinement.optim import pose_optimization_least_squares_with_t
from utils.drawing_utils import *

def set_equal_axis_ranges(fig, xs, ys, zs, pad=0.05):
    """
    Forces x,y,z axes to all cover the same span (plus optional padding),
    which in turn makes one unit on X the same visual length as one unit on Y or Z.
    """
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    pad_amt = span * pad
    midx = (xmin + xmax) / 2
    midy = (ymin + ymax) / 2
    midz = (zmin + zmax) / 2
    
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[midx - span/2 - pad_amt, midx + span/2 + pad_amt]),
            yaxis=dict(range=[midy - span/2 - pad_amt, midy + span/2 + pad_amt]),
            zaxis=dict(range=[midz - span/2 - pad_amt, midz + span/2 + pad_amt])
        )
    )

pv_fifo = deque(maxlen=10)
path = r''
name_of_run = r''
json_filename = 'personal_video.json'
calib_filename = '1280_720_calibration.pkl'
calibration_path = 'calibrations'
host = None

# get the model weight and checkpoint path SAM2 and Yolo model
sam2_model_path = ""
sam2_config_path = ""
yolo_model_path = ""

# get the calibration data for the left and right cameras
port_left = hl2ss.StreamPort.RM_VLC_LEFTFRONT
calibration_lf = hl2ss_3dcv.get_calibration_rm(calibration_path, host, port_left)
rotation_lf = hl2ss_3dcv.rm_vlc_get_rotation(port_left)
K_left_raw, pose_left_raw = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_lf.intrinsics, calibration_lf.extrinsics, rotation_lf)
K_left_4x4 = K_left_raw.T
pose_left_4x4 = pose_left_raw.T

port_right = hl2ss.StreamPort.RM_VLC_RIGHTFRONT
calibration_rf = hl2ss_3dcv.get_calibration_rm(calibration_path, host, port_right)
rotation_rf = hl2ss_3dcv.rm_vlc_get_rotation(port_right)
K_right_raw, pose_right_raw = hl2ss_3dcv.rm_vlc_rotate_calibration(calibration_rf.intrinsics, calibration_rf.extrinsics, rotation_rf)
K_right_4x4 = K_right_raw.T
pose_right_4x4 = pose_right_raw.T

json_data = None
with open(os.path.join(path, json_filename), 'r') as f:
    json_data = json.load(f)

print(memsize(json_data))
keyframes_backwards = {}
index = 0
while True:
    if (str(index) in json_data["keyframes"]):
        keyframes_backwards[json_data["keyframes"][str(index)]] = index
    else:
        break
    index+= 1

calibrationData = None
with open(os.path.join(calib_filename), 'rb') as f:
    calibrationData = pickle.load(f)
print(calibrationData)

intrinsics_opencv, extrinsics_opencv = PVCalibrationToOpenCVFormat(calibrationData)

if __name__ == '__main__':
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    seq_pv = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))
    seq_vlc_left = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_VLC_LEFTFRONT)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))
    seq_vlc_right = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_VLC_RIGHTFRONT)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))

    first_try = True
    seq_pv.open()
    rd_pv = seq_pv.get_reader()
    tip_local = np.array([[-0.03303],
                      [-0.5700],
                      [ -0.0275     ]], dtype=np.float32)
    tip_positions = []

    seq_vlc_left.open()
    seq_vlc_right.open()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # get the YOLO model
    model_path = yolo_model_path
    model   = YOLO(model_path)
    # get the SAM2 tracker
    tracker = SAM2PointTracker(
        config_file=sam2_config_path,
        ckpt_path=sam2_model_path,
        device=device,
        num_objects=1,
        frame_rate=30,
        camera_type="rgb"
    )
    tracker_gray_left = SAM2PointTracker(
        config_file=sam2_config_path,
        ckpt_path=sam2_model_path,
        device=device,
        num_objects=1,
        frame_rate=30,
        camera_type="gray_left"
    )
    tracker_gray_right = SAM2PointTracker(
        config_file=sam2_config_path,
        ckpt_path=sam2_model_path,
        device=device,
        num_objects=1,
        frame_rate=30,
        camera_type="gray_right"
    )   
    # start the main loop
    while ((cv2.waitKey(1) & 0xFF) != 27):
        packet = rd_pv.get_next_packet()
        if (packet is None):
            print('End of PV file')
            break
        _, left_vlc_packet = seq_vlc_left.get_next_packet(packet.timestamp)
        _, right_vlc_packet = seq_vlc_right.get_next_packet(packet.timestamp)
        pv_fifo.append(packet)
        # undistort and rotate the left and right images
        lf_u = hl2ss_3dcv.rm_vlc_undistort(left_vlc_packet.payload.image, calibration_lf.undistort_map)
        left_image_undist_rotated = hl2ss_3dcv.rm_vlc_rotate_image(lf_u, rotation_lf)
        rf_u = hl2ss_3dcv.rm_vlc_undistort(right_vlc_packet.payload.image, calibration_rf.undistort_map)
        right_image_undist_rotated = hl2ss_3dcv.rm_vlc_rotate_image(rf_u, rotation_rf)
        # if the packet is a keyframe, print the keyframe information
        if (packet.timestamp in keyframes_backwards):
            print("keyframe: ", keyframes_backwards[packet.timestamp], "timestamp: ", packet.timestamp)
            print("")
            print(f'Frame captured at {packet.timestamp}')
            print(f'Focal length: {packet.payload.focal_length}')
            print(f'Principal point: {packet.payload.principal_point}')
            print(f'Exposure Time: {packet.payload.exposure_time}')
            print(f'Exposure Compensation: {packet.payload.exposure_compensation}')
            print(f'Lens Position (Focus): {packet.payload.lens_position}')
            print(f'Focus State: {packet.payload.focus_state}')
            print(f'ISO Speed: {packet.payload.iso_speed}')
            print(f'White Balance: {packet.payload.white_balance}')
            print(f'ISO Gains: {packet.payload.iso_gains}')
            print(f'White Balance Gains: {packet.payload.white_balance_gains}')
            print(f'Resolution {packet.payload.resolution}')
            print(f'Pose')
            print(packet.pose)
            right_image = cv2.cvtColor(right_image_undist_rotated, cv2.COLOR_GRAY2BGR)
            left_image = cv2.cvtColor(left_image_undist_rotated, cv2.COLOR_GRAY2BGR)
            annotated_image, selected_point, three_3d_lines, rvec, tvec, K, obj_pts, img_pts = process_image(packet.payload.image, intrinsics_opencv, instrument_type, model)
            pose_to_save = np.zeros_like((4,4))
            R_init, _ = cv2.Rodrigues(rvec)
            tip_world = R_init @ tip_local + tvec
            pose_init = np.vstack([np.hstack([R_init, tvec]), np.array([0,0,0,1.0])])
            if annotated_image is not None:
                shaft = np.array([[-0.03303, -0.03094, -0.0275],
                    [-0.03303, -0.170, -0.0275]], dtype=np.float32)
                shaft_pts = np.vstack((np.column_stack(shaft), np.ones(shaft.shape[0])))
                # project the shaft points onto the left and right frames
                pts_in_left_frame = pose_left_4x4 @ np.linalg.inv(extrinsics_opencv) @ pose_init @ shaft_pts
                pts_in_right_frame = pose_right_4x4 @ np.linalg.inv(extrinsics_opencv) @ pose_init @ shaft_pts
                img_pts_left = K_left_4x4 @ pts_in_left_frame
                img_pts_left /= img_pts_left[2,:]
                img_pts_right = K_right_4x4 @ pts_in_right_frame
                img_pts_right /= img_pts_right[2,:]
                # process the image with the SAM2 tracker
                out, three_2d_lines = tracker.process(
                    packet.payload.image, point_prompt=selected_point
                )
                out_left, center_line_left = tracker_gray_left.process(
                    left_image, point_prompt=img_pts_left
                )
                out_right, center_line_right = tracker_gray_right.process(
                    right_image, point_prompt=img_pts_right
                )
                # refine the pose using the pose optimization
                if three_2d_lines is not None:
                    result, rvec_corrected, tvec_corrected = pose_optimization_least_squares_with_t(
                        rvec=rvec, 
                        tvec=tvec,
                        K=K, 
                        target_img_lines = three_2d_lines,
                        point3d = three_3d_lines,
                        obj_pts_square = obj_pts,
                        img_pts_square = img_pts,
                        test_img=packet.payload.image,
                        rgb_extrinsics=extrinsics_opencv,
                        left_camera_calibration=(K_left_4x4, pose_left_4x4),
                        right_camera_calibration=(K_right_4x4, pose_right_4x4),
                        left_shaft_line=center_line_left,
                        right_shaft_line=center_line_right,
                    )
                    rvec_corrected = rvec_corrected.reshape(3, 1)
                    tvec_corrected = tvec_corrected.reshape(3, 1)
                    R_corr, _ = cv2.Rodrigues(rvec_corrected)
                    tip_world = R_corr @ tip_local + tvec_corrected
                    pose_to_save = np.vstack([np.hstack([R_corr, tvec_corrected]), np.array([0,0,0,1.0])])
                    tip_positions.append(tip_world.ravel())
                    # show the result
                    cv2.imshow("result", result)
                    cv2.waitKey(1)
                else:
                    rvec_corrected, tvec_corrected = rvec, tvec
                raw_data_output_name = os.path.join("/home/arssist/AR_proj_hk/output", name_of_run, "raw_output")
                os.makedirs(raw_data_output_name, exist_ok=True)
                write_index = keyframes_backwards[packet.timestamp]
                np.savetxt(os.path.join(raw_data_output_name, f"{write_index:05d}.txt"), pose_to_save)
                print("NAME OF RUN")
                print(name_of_run)
    # plot the tip positions
    if tip_positions:
        xs, ys, zs = zip(*tip_positions)
        color_indices = list(range(len(xs)))
        N = len(xs)
        labels = [str(i) for i in range(N)]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode='markers+text',
                    text=labels,
                    textposition='top center',
                    marker=dict(
                        size=5,
                        color=color_indices,
                        colorscale='Rainbow',
                        cmin=0,
                        cmax=N-1,
                        showscale=False
                    ),
                    hovertemplate=(
                        "Index %{text}<br>"
                        "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                        "<extra></extra>"
                    )
                )
            ]
        )
        x_all = np.array(xs)
        y_all = np.array(ys)
        z_all = np.array(zs)
        set_equal_axis_ranges(fig, x_all, y_all, z_all)
        fig.show()
    rd_pv.close()
