import copy
from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm  # Thêm thư viện tqdm tạo progress bar
from pprint import pprint
import csv
from numbers import Real
from datetime import datetime
import re

from pose_pipeline.smpl_runner import create_smpl_model
from models.keypoints_map import load_keypoints3d_map
#from config_loader import resolve_inputs
from pose_pipeline.smpl_runner import get_3d_joints_for_frame
from utils.json_io import write_json
from utils.compat import load_joblib_compat, resolve_person_payload
from belief_calculator.correction import calculate_camera_belief
#from preprocess_pipeline.calib import resolve_selected_offset_from_camera_profile
import pdb

POSE_OUTPUT_SUBDIRS = ("keypoints3d", "metadata")
MAPPING = {"global_belief" : "GBelief", "local_belief" : "LBelief", "visibility" : "Vis.",}

def _extract_person_payload(data):
    return resolve_person_payload(data)

def _to_python(value):
    """Chuyển NumPy arrays/scalars về kiểu dữ liệu Python."""
    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        return value.item()

    return value

def _flatten_dict(data, prefix=""):
    """
    Chuyển dictionary lồng nhau thành dictionary phẳng.

    Vector 3D được tách thành các trường x, y, z.
    Các list khác được đánh chỉ số.
    """
    result = {}

    data = _to_python(data)

    if isinstance(data, dict):
        for key, value in data.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_dict(value, field))

    elif isinstance(data, (list, tuple)):
        values = [_to_python(v) for v in data]

        is_vector3 = (
            len(values) == 3
            and all(
                isinstance(v, Real) and not isinstance(v, bool)
                for v in values
            )
        )

        if is_vector3:
            for axis, value in zip(("x", "y", "z"), values):
                result[f"{prefix}.{axis}"] = value
        else:
            for index, value in enumerate(values):
                field = f"{prefix}.{index}"
                result.update(_flatten_dict(value, field))

    else:
        result[prefix] = data

    return result

def load_pkl_data(file_path, target_person=None):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("PKL file not found: {}".format(file_path))
    person_data = resolve_person_payload(load_joblib_compat(file_path), pkl_path=file_path, target_person=target_person)
    if person_data is None:
        raise ValueError("Cannot read person payload from {}".format(file_path))
    for key in ("pose", "trans", "betas"):
        if key not in person_data:
            raise KeyError("Missing key {!r} in {}. Available keys: {}".format(key, file_path, list(person_data.keys())))
    return person_data


def slice_person_frames(person_data, start, frame_count):
    start = int(start)
    frame_count = int(frame_count)
    total_frames = int(np.asarray(person_data["pose"]).shape[0])
    end = min(total_frames, start + frame_count)
    out = copy.deepcopy(dict(person_data))
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            out[key] = value[start:end].copy()
    return out


def _shared_betas(*person_data: dict) -> np.ndarray:
    betas = []
    for person in person_data:
        values = np.asarray(person["betas"], dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected non-empty betas with shape (frames, dims), got {values.shape}")
        betas.append(values)
    if len({values.shape[1] for values in betas}) != 1:
        raise ValueError("Camera beta dimensions do not match")
    return np.median(np.concatenate(betas, axis=0), axis=0).astype(np.float32)


def _resolve_tracking_frame_ids(person_data):
    tracking = person_data.get("tracking_results_for_reproj")
    if not isinstance(tracking, dict) or "frame_id" not in tracking:
        return None

    frame_ids = np.asarray(tracking["frame_id"]).astype(int).reshape(-1)
    total_frames = int(np.asarray(person_data["pose"]).shape[0])
    if frame_ids.shape[0] != total_frames:
        print(
            "[Pose] Tracking frame_id length mismatch with pose frames: "
            f"{frame_ids.shape[0]} vs {total_frames}. Falling back to sequential indices."
        )
        return None
    return frame_ids.tolist()


def _clean_pose_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob("pose_data_*.json"):
        old_json.unlink(missing_ok=True)
    for subdir in POSE_OUTPUT_SUBDIRS:
        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for old_json in target_dir.glob("pose_data_*.json"):
            old_json.unlink(missing_ok=True)


def flatten_belief_data(belief_data: dict, mapping = MAPPING) -> dict:
    """Chuyển belief của từng joint thành các cột CSV."""
    result = {}
    

    for joint_name, values in belief_data.items():
        for attribute in (
            "global_belief",
            "local_belief",
            "visibility",
        ):
            result[f"{joint_name}.{mapping[attribute]}"] = (
                values.get(attribute)
            )

    return result

def _split_mpi_inf_3dhpe(cam_file):
    elems = cam_file.split("/")
    index = elems.index("imageSequences")
    set_name = elems[index - 2] + "/" + elems[index-1]
    match = re.search(r'wham_(?:cam|video_)?(.*?)\.pkl', elems[-1])
    x_value = None
    if match:
        x_value = match.group(1) # Lấy ra giá trị X
        camera_name = "cam" + str(x_value)
    else:
        camera_name = elems[-1]
    return set_name, camera_name, x_value


def _create_pose_output_dirs(output_dir):
    """Tạo các thư mục đầu ra."""
    directories = {
        "keypoints": output_dir / "keypoints3d",
        "metadata": output_dir / "metadata",
        "belief": output_dir / "belief",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def _prepare_pose_export_context(args, config):
    """Khởi tạo model, dữ liệu PKL và thông tin camera."""
    cam1_file = Path(args.pkl)
    output_dir = Path(args.output)
    set_name, camera_name, _ = _split_mpi_inf_3dhpe(args.pkl)

    directories = _create_pose_output_dirs(output_dir)

    print("[Pose] Loading SMPL model...")
    model = create_smpl_model(Path(args.smpl))

    print(f"[Pose] Loading camera: {cam1_file} (target_person=1)")
    cam1_data = load_pkl_data(cam1_file, target_person="1")
    tracking_ids = _resolve_tracking_frame_ids(cam1_data)

    cam1_start = 0
    min_frames = len(cam1_data["pose"])
    print(f"[Pose] Camera frames: {min_frames}")

    sync_result = {
        "offset": 0,
        "left_start": cam1_start,
        "source_pkls": {camera_name: camera_name},
    }

    export_data = slice_person_frames(cam1_data, cam1_start, min_frames)

    map_path = "configs/keypoints3D_map.yml"
    map_data = load_keypoints3d_map(map_path)
    joint_names = [kp["name"] for kp in map_data["keypoints"]]

    return {
        "model": model,
        "cam1_data": cam1_data,
        "export_data": export_data,
        "tracking_ids": tracking_ids,
        "cam1_start": cam1_start,
        "min_frames": min_frames,
        "set_name": set_name,
        "camera_name": camera_name,
        "sync_result": sync_result,
        "joint_names": joint_names,
        "map_path": map_path,
        "output_dir": output_dir,
        "directories": directories,
        "use_world": bool((config or {}).get("use_world_coords", False)),
    }


def _get_source_frame_id(tracking_ids, start, index):
    """Lấy frame ID gốc từ tracking hoặc chỉ số tuần tự."""
    if tracking_ids is not None:
        return int(tracking_ids[start + index])
    return start + index


def _calculate_frame_pose(context, index, alpha, beta, tau):
    """Tính 3D joints và camera belief cho một frame."""
    export_data = context["export_data"]

    joints = get_3d_joints_for_frame(
        context["model"],
        export_data,
        index,
        "src/ray_casting/models/J_regressor_body25_plus_palm27.npy",
        context["map_path"],
        use_world=context["use_world"],
    )

    verts = export_data["verts"][index]

    belief = calculate_camera_belief(
        joints=joints,
        verts=verts,
        joint_names=context["joint_names"],
        alpha=alpha,
        beta=beta,
        occlusion_tau=tau,
    )

    return joints, belief


def _build_frame_metadata(context, source_frame_id):
    """Tạo metadata cho một frame."""
    camera_name = context["camera_name"]

    return {
        "metadata": {
            "camera_sync": context["sync_result"],
            "source_frame_indices": {
                camera_name: source_frame_id,
            },
        }
    }


def _write_frame_jsons(context, frame_id, joints, metadata, belief):
    """Ghi ba loại JSON cho một frame."""
    directories = context["directories"]
    camera_name = context["camera_name"]
    cam1_data = context["cam1_data"]

    pkl_frame_id = int(cam1_data["frame_ids"][frame_id - 1])

    belief_output = {
        "camera": camera_name,
        "source_frame_id": metadata["metadata"][
            "source_frame_indices"
        ][camera_name],
        "pkl_frame_id": pkl_frame_id,
        "joints": belief,
    }

    write_json(
        directories["keypoints"] / f"pose_data_{frame_id}.json",
        joints,
    )
    write_json(
        directories["metadata"] / f"pose_data_{frame_id}.json",
        metadata,
    )
    write_json(
        directories["belief"] / f"belief_{frame_id}.json",
        belief_output,
    )

    return pkl_frame_id


def _build_pose_csv_row(context, frame_id, pkl_frame_id,
                        source_frame_id, joints, metadata, belief):
    """Tổng hợp dữ liệu của một frame thành một dòng CSV."""
    camera_name = context["camera_name"]
    sync = context["sync_result"]

    row = {
        "F_ID": frame_id,
        "Set": context["set_name"],
        "Cam": camera_name,
        "PKL_F_ID": pkl_frame_id,
        "offset": sync["offset"],
        "left_start": sync["left_start"],
        "PKLs": sync["source_pkls"][camera_name],
        "FrameIndices": source_frame_id,
    }

    row.update(_flatten_dict(joints))
    row.update(flatten_belief_data(belief))

    return row


def _process_pose_frame(context, index, alpha, beta, tau):
    """Xử lý hoàn chỉnh một frame."""
    frame_id = index + 1

    joints, belief = _calculate_frame_pose(
        context, index, alpha, beta, tau
    )

    source_frame_id = _get_source_frame_id(
        context["tracking_ids"],
        context["cam1_start"],
        index,
    )

    metadata = _build_frame_metadata(
        context, source_frame_id
    )

    pkl_frame_id = _write_frame_jsons(
        context, frame_id, joints, metadata, belief
    )

    return _build_pose_csv_row(
        context,
        frame_id,
        pkl_frame_id,
        source_frame_id,
        joints,
        metadata,
        belief,
    )


def _build_pose_csv_fieldnames(joint_names):
    """Tạo danh sách cột CSV theo thứ tự YAML."""
    fieldnames = [
        "F_ID", "Set", "Cam", "PKL_F_ID",
        "offset", "left_start", "PKLs", "FrameIndices",
    ]

    for joint_name in joint_names:
        for axis in ("x", "y", "z"):
            fieldnames.append(f"{joint_name}.{axis}")

    for joint_name in joint_names:
        for attribute in (
            "global_belief",
            "local_belief",
            "visibility",
        ):
            fieldnames.append(
                f"{joint_name}.{MAPPING[attribute]}"
            )

    return fieldnames


def _validate_pose_csv_rows(csv_rows, fieldnames):
    """Kiểm tra sự tương ứng giữa dữ liệu và các cột CSV."""
    expected = set(fieldnames)

    for index, row in enumerate(csv_rows):
        missing = expected - set(row)
        extra = set(row) - expected

        if missing or extra:
            raise ValueError(
                f"CSV row {index + 1} has invalid keys. "
                f"Missing: {sorted(missing)}; "
                f"Extra: {sorted(extra)}"
            )


def _write_pose_csv(csv_rows, fieldnames, output_dir):
    """Kiểm tra và ghi dữ liệu CSV."""
    csv_path = output_dir / "pose_export.csv"

    _validate_pose_csv_rows(csv_rows, fieldnames)

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    return csv_path


def run_pose_export(
    args, belief_alpha=0.1, belief_beta=0.8, occlusion_tau=0.05,
    config=None,
) -> None:
    """Export 3D pose, metadata và belief ra JSON/CSV."""

    context = _prepare_pose_export_context(args, config)
    csv_rows = []

    for index in tqdm(
        range(context["min_frames"]),
        desc="[Pose] Exporting JSONs and CSV",
        unit="frame",
    ):
        row = _process_pose_frame(
            context,
            index,
            belief_alpha,
            belief_beta,
            occlusion_tau,
        )
        csv_rows.append(row)

    fieldnames = _build_pose_csv_fieldnames(
        context["joint_names"]
    )

    csv_path = _write_pose_csv(
        csv_rows,
        fieldnames,
        context["output_dir"],
    )

    now = datetime.now()
    # Định dạng chuỗi hiển thị: Ngày/Tháng/Năm Giờ:Phút:Giây
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")

    print(f"[Pose] {current_time} Done. Output: {context['output_dir']}")
    print(f"[Pose] {current_time} CSV exported: {csv_path}")
    print(f"[Pose] {current_time} Frames exported: {len(csv_rows)}")
    print(f"[Pose] {current_time} CSV columns: {len(fieldnames)}")

"""## 8. Fusion phase"""