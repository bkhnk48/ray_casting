#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export annot.mat (MPI-INF-3DHP) to CSV.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Giả định module này tồn tại trong source của bạn
from pose_pipeline.executor import _split_mpi_inf_3dhpe

JOINT_NAMES = [
    "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]

JOINT_INDICES = [5, 9, 14, 10, 15, 11, 16, 18, 23, 19, 24, 20, 25]


def matlab_cells(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    arr = np.asarray(value)
    if arr.dtype == object:
        return list(arr.reshape(-1))
    return [value]


def annotation_matrix(value, name):
    arr = np.squeeze(np.asarray(value))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: cần ma trận 2D, nhận được shape={arr.shape}")
    return arr


def expand_frames(frame_value, n_cameras):
    cells = matlab_cells(frame_value)
    if len(cells) == 1:
        shared = np.asarray(cells[0]).squeeze().reshape(-1)
        return [shared] * n_cameras
    if len(cells) == n_cameras:
        return [np.asarray(x).squeeze().reshape(-1) for x in cells]
    raise ValueError(f"Không thể ghép frames. Có {len(cells)} cells.")


def expand_camera_ids(camera_value, n_cameras):
    if camera_value is None:
        return list(range(n_cameras))
    cells = matlab_cells(camera_value)
    if len(cells) == 1:
        arr = np.asarray(cells[0]).squeeze().reshape(-1)
        if arr.size == n_cameras:
            return [int(x) for x in arr]
    if len(cells) == n_cameras:
        result = []
        for i, item in enumerate(cells):
            arr = np.asarray(item).squeeze().reshape(-1)
            result.append(int(arr[0]))
        return result
    raise ValueError("Không thể xác định camera IDs từ annot.mat.")


def create_csv_columns():
    columns = ["video", "camera", "frame"]
    for joint in JOINT_NAMES:
        columns.extend([f"{joint}_{axis}" for axis in ("x", "y", "z")])
    return columns


def load_and_validate_mat(mat_path):
    try:
        mat = loadmat(mat_path, simplify_cells=True)
    except Exception:
        raise KeyError("Không tìm thấy annot.mat tại đường dẫn cung cấp.")
    
    if "annot3" not in mat or "frames" not in mat:
        raise KeyError("Thiếu biến annot3 hoặc frames trong annot.mat.")
    return mat


def process_single_camera(writer, camera_id, annot_cell, frames_cell, video_prefix, frame_limit):
    ann = annotation_matrix(annot_cell, f"Camera {camera_id}")
    if ann.shape[1] % 3 != 0:
        raise ValueError(f"Camera {camera_id} có số cột không chia hết cho 3.")
    
    joint_count = ann.shape[1] // 3
    if max(JOINT_INDICES) >= joint_count:
        raise ValueError(f"Dữ liệu camera {camera_id} chỉ có {joint_count} joints.")

    n_frames = min(ann.shape[0], len(frames_cell))
    if frame_limit is not None:
        n_frames = min(n_frames, frame_limit)

    poses = ann[:n_frames].reshape(n_frames, joint_count, 3)
    frame_numbers = frames_cell[:n_frames]

    for f_idx in range(n_frames):
        row = [video_prefix, camera_id, int(frame_numbers[f_idx])]
        for j_idx in JOINT_INDICES:
            x, y, z = poses[f_idx][j_idx]
            row.extend([float(x), float(y), float(z)])
        writer.writerow(row)
    
    return n_frames


def write_all_cameras(output_path, mat_data, args, index_of_camera, video_prefix):
    annot_cells = matlab_cells(mat_data["annot3"])
    n_cameras = len(annot_cells)
    frame_cells = expand_frames(mat_data["frames"], n_cameras)
    camera_ids = expand_camera_ids(mat_data.get("cameras"), n_cameras)

    total_rows, exported_cameras = 0, 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(create_csv_columns())

        for c_idx in range(n_cameras):
            cam_id = camera_ids[c_idx]
            if index_of_camera is not None and cam_id != int(index_of_camera):
                continue
            
            rows = process_single_camera(
                writer, cam_id, annot_cells[c_idx], frame_cells[c_idx], 
                video_prefix, getattr(args, 'frame_count', None)
            )
            total_rows += rows
            exported_cameras += 1
            
    return exported_cameras, total_rows


def run_annot_mat_export(args):
    print(f"Đọc file: {args.ground_truth}")
    mat_data = load_and_validate_mat(args.ground_truth)
    set_name, camera_name, index_of_camera = _split_mpi_inf_3dhpe(args.pkl)

    output_path = Path(args.output) / f"{set_name}_{camera_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_prefix = f"{set_name}_{camera_name}"

    cams, rows = write_all_cameras(
        output_path, mat_data, args, index_of_camera, video_prefix
    )

    print("\n" + "=" * 50 + "\nHOÀN THÀNH\n" + "=" * 50)
    print(f"File CSV: {output_path}")
    print(f"Số camera được xuất: {cams} | Tổng số frames: {rows}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export MPI-INF-3DHP to CSV.")
    parser.add_argument("--ground_truth", required=True, help="Đường dẫn tới annot.mat")
    parser.add_argument("--pkl", required=True, help="Đường dẫn tới file PKL")
    parser.add_argument("--video", required=True, help="Tên video sequence")
    parser.add_argument("--output", default="outputs", help="Thư mục đầu ra")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    
    args = parser.parse_args()
    if args.frame_count is not None and args.frame_count <= 0:
        parser.error("--frame-count phải lớn hơn 0")
    return args


def main():
    args = parse_arguments()
    try:
        run_annot_mat_export(args)
    except Exception as exc:
        print(f"\nLỖI: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()