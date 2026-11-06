#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export annot.mat (MPI-INF-3DHP) to CSV.

- Đọc annot3, KHÔNG đọc univ_annot3.
- annot3 chứa tọa độ 3D trong hệ tọa độ camera.
- Hỗ trợ annot3 có nhiều MATLAB cells (ví dụ 14 cameras).
- Hỗ trợ frames là một vector dùng chung cho tất cả cameras.
- Xuất 13 joints, mỗi joint có tọa độ x, y, z.
- Joint indices phải được xác định theo mpii_get_joint_set.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import pdb

# Thứ tự 13 joints trong CSV.
JOINT_NAMES = [
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# 13 joints, zero-based indices in the standardized 28-joint order
JOINT_INDICES = [
    5,   # neck
    9,   # left_shoulder
    14,  # right_shoulder
    10,  # left_elbow
    15,  # right_elbow
    11,  # left_wrist
    16,  # right_wrist
    18,  # left_hip
    23,  # right_hip
    19,  # left_knee
    24,  # right_knee
    20,  # left_ankle
    25,  # right_ankle
]


def matlab_cells(value):
    """
    Chuyển dữ liệu MATLAB cell thành Python list.

    Với simplify_cells=True, scipy có thể trả về:
    - list
    - tuple
    - numpy array dtype=object
    - numpy array thông thường
    """

    if isinstance(value, (list, tuple)):
        return list(value)

    arr = np.asarray(value)

    if arr.dtype == object:
        return list(arr.reshape(-1))

    return [value]


def annotation_matrix(value, name):
    """Chuyển một annotation cell thành ma trận 2D."""

    arr = np.asarray(value)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(
            f"{name}: cần ma trận 2D, nhưng nhận được shape={arr.shape}"
        )

    return arr


def expand_frames(frame_value, n_cameras):
    """
    Xử lý hai trường hợp:

    Trường hợp 1:
        frames có một vector dùng chung cho tất cả camera.

    Trường hợp 2:
        frames có một vector riêng cho mỗi camera.
    """

    cells = matlab_cells(frame_value)

    if len(cells) == 1:
        shared_frames = np.asarray(cells[0]).squeeze().reshape(-1)

        return [shared_frames] * n_cameras

    if len(cells) == n_cameras:
        return [
            np.asarray(x).squeeze().reshape(-1)
            for x in cells
        ]

    raise ValueError(
        f"Không thể ghép frames với annot3: "
        f"frames có {len(cells)} cell, "
        f"annot3 có {n_cameras} cell."
    )


def expand_camera_ids(camera_value, n_cameras):
    """
    Đọc camera IDs.

    Có thể gặp:
    - Một vector [0, 1, 2, ..., 13].
    - Một MATLAB cell chứa 14 camera IDs.
    - Không có biến cameras.
    """
    if camera_value is None:
        print("Cảnh báo: annot.mat không có biến cameras. Tạm sử dụng chỉ số cell làm camera ID.")
        return list(range(n_cameras))

    cells = matlab_cells(camera_value)

    # Một vector chứa toàn bộ camera IDs.
    if len(cells) == 1:
        arr = np.asarray(cells[0]).squeeze().reshape(-1)
        if arr.size == n_cameras:
            return [int(x) for x in arr]

    # Một cell cho mỗi camera.
    if len(cells) == n_cameras:
        result = []

        for i, item in enumerate(cells):
            arr = np.asarray(item).squeeze().reshape(-1)

            if arr.size == 0:
                raise ValueError(f"Camera ID tại cell {i} bị rỗng.")    
            result.append(int(arr[0]))
        return result

    raise ValueError(f"Không thể xác định camera IDs: có {len(cells)} phần tử nhưng annot3 có {n_cameras} cell.")





def create_csv_columns():
    columns = [
        "video",
        "camera",
        "frame",
    ]

    for joint in JOINT_NAMES:
        columns.extend([
            f"{joint}_{axis}"
            for axis in ("x", "y", "z")
        ])

    return columns


def export(args):
    print(f"Đọc file: {args.mat}")
    mat = loadmat(
        args.mat,
        simplify_cells=True
    )
    pdb.set_trace()

    # Kiểm tra các biến bắt buộc.
    if "annot3" not in mat:
        raise KeyError(
            "Không tìm thấy biến annot3 trong annot.mat."
        )

    if "frames" not in mat:
        raise KeyError(
            "Không tìm thấy biến frames trong annot.mat."
        )

    # Đọc annotation cells.
    annot_cells = matlab_cells(mat["annot3"])

    n_cameras = len(annot_cells)

    print(f"Số annotation cells: {n_cameras}")

    # Ghép frames với annotation cells.
    frame_cells = expand_frames(
        mat["frames"],
        n_cameras
    )

    # Đọc camera IDs.
    camera_ids = expand_camera_ids(
        mat.get("cameras"),
        n_cameras
    )

    # Kiểm tra joint indices.
    joint_indices = JOINT_INDICES

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_rows = 0
    exported_cameras = 0

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(create_csv_columns())

        for cell_idx in range(n_cameras):

            camera_id = camera_ids[cell_idx]

            # Nếu chỉ định camera thì bỏ qua các camera khác.
            if (
                args.camera is not None
                and camera_id != args.camera
            ):
                continue

            ann = annotation_matrix(
                annot_cells[cell_idx],
                f"annot3 cell {cell_idx}"
            )

            frames = frame_cells[cell_idx]

            print(
                f"\nCamera ID: {camera_id}"
            )

            print(
                f"Annotation shape: {ann.shape}"
            )

            print(
                f"Frames shape: {frames.shape}"
            )

            # annot3 có dạng:
            # x1,y1,z1,x2,y2,z2,...,x28,y28,z28

            if ann.shape[1] % 3 != 0:
                raise ValueError(
                    f"annot3 cell {cell_idx} có "
                    f"{ann.shape[1]} cột, không chia hết cho 3."
                )

            joint_count = ann.shape[1] // 3

            print(
                f"Số joints trong annot3: {joint_count}"
            )

            if max(joint_indices) >= joint_count:
                raise ValueError(
                    f"Joint index lớn nhất là {max(joint_indices)}, "
                    f"nhưng dữ liệu chỉ có {joint_count} joints."
                )

            # Chỉ lấy số frame có cả annotation và frame number.
            n_frames = min(
                ann.shape[0],
                len(frames)
            )

            # Nếu người dùng chỉ định frame count.
            if args.frame_count is not None:
                n_frames = min(
                    n_frames,
                    args.frame_count
                )

            print(
                f"Số frames sẽ xuất: {n_frames}"
            )

            # Reshape:
            # (F, 84) -> (F, 28, 3)
            poses = ann[:n_frames].reshape(
                n_frames,
                joint_count,
                3
            )

            frame_numbers = frames[:n_frames]

            # Xuất từng frame.
            for frame_idx in range(n_frames):

                row = [
                    args.video,
                    camera_id,
                    int(frame_numbers[frame_idx])
                ]

                pose = poses[frame_idx]

                # Xuất đúng thứ tự 13 joints.
                for joint_idx in joint_indices:

                    x, y, z = pose[joint_idx]

                    row.extend([
                        float(x),
                        float(y),
                        float(z)
                    ])

                writer.writerow(row)

                total_rows += 1

            exported_cameras += 1

    print("\n" + "=" * 50)
    print("HOÀN THÀNH")
    print("=" * 50)

    print(f"File CSV: {output_path}")
    print(f"Số camera được xuất: {exported_cameras}")
    print(f"Tổng số frames: {total_rows}")
    print(f"Số joints mỗi frame: 13")
    print(f"Số tọa độ mỗi frame: 39")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Export 13 camera-coordinate 3D joints "
            "from MPI-INF-3DHP annot.mat to CSV."
        )
    )

    parser.add_argument(
        "--mat",
        required=True,
        help="Đường dẫn tới annot.mat"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Tên video hoặc sequence"
    )

    parser.add_argument(
        "--output",
        default="annot3_13joints.csv",
        help="Đường dẫn CSV đầu ra"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Chỉ xuất camera ID cụ thể. Mặc định xuất tất cả."
    )

    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="Số frame tối đa cần xuất."
    )

    
    args = parser.parse_args()

    if args.frame_count is not None:
        if args.frame_count <= 0:
            parser.error("--frame-count phải lớn hơn 0")

    try:
        export(args)

    except Exception as exc:
        print(
            f"\nLỖI: {exc}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
