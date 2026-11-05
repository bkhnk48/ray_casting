import copy
from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm  # Thêm thư viện tqdm tạo progress bar
from pprint import pprint
import csv
from numbers import Real

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
    camera_name = elems[-1]
    return set_name, camera_name

#def run_pose_export(cam1_pkl, config = None) -> None:
def run_pose_export(
    args,
    #camera_name="camera1",
    belief_alpha=0.1,
    belief_beta=0.8,
    occlusion_tau=0.05,
    config=None,
) -> None:
    pose_cfg = config or {}
    cam1_pkl_path = args.pkl
    outputs_path = args.output

    cam1_file = Path(cam1_pkl_path)
    set_name, camera_name = _split_mpi_inf_3dhpe(cam1_pkl_path)

    smpl_model_path = Path("src/ray_casting/models/SMPL_NEUTRAL.pkl")
    output_dir = Path(outputs_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    keypoints_dir = output_dir / "keypoints3d"
    metadata_dir = output_dir / "metadata"
    belief_dir = output_dir / "belief"

    for directory in (keypoints_dir, metadata_dir, belief_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("[Pose] Loading SMPL model...")
    model = create_smpl_model(smpl_model_path)

    target_person = "1"

    print(f"[Pose] Loading camera 1: {cam1_file} (target_person={target_person})")

    cam1_data = load_pkl_data(cam1_file, target_person=target_person,)

    cam1_tracking_frame_ids = _resolve_tracking_frame_ids(cam1_data)

    cam1_start = 0

    #sync_result = { "offset": 0, "left_start": cam1_start, "source_pkls": { camera_name: str(cam1_file.resolve()),}, }
    sync_result = { "offset": 0, "left_start": cam1_start, "source_pkls": { camera_name: camera_name,}, }

    min_frames = len(cam1_data["pose"])

    print(f"[Pose] Cam1 frames: {min_frames}")

    j_regressor_path = ("src/ray_casting/models/J_regressor_body25_plus_palm27.npy")

    cam1_export_data = slice_person_frames(cam1_data, cam1_start, min_frames,)

    use_world = bool(pose_cfg.get("use_world_coords", False))

    path_keypoints3d_map = "configs/keypoints3D_map.yml"

    map_data = load_keypoints3d_map(path_keypoints3d_map)

    joint_names = [ kp["name"] for kp in map_data["keypoints"] ]

    csv_rows = []

    for i in tqdm(range(min_frames), desc="[Pose] Exporting JSONs and CSV", unit="frame",):
        out_frame_id = i + 1

        joints = get_3d_joints_for_frame(
            model,
            cam1_export_data,
            i,
            j_regressor_path,
            path_keypoints3d_map,
            use_world=use_world,
        )

        verts = cam1_export_data["verts"][i]

        belief_data = calculate_camera_belief(
            joints=joints,
            verts=verts,
            joint_names=joint_names,
            alpha=belief_alpha,
            beta=belief_beta,
            occlusion_tau=occlusion_tau,
        )

        source_frame_id = (
            int(cam1_tracking_frame_ids[cam1_start + i])
            if cam1_tracking_frame_ids is not None
            else cam1_start + i
        )

        pkl_frame_id = int(cam1_data["frame_ids"][i])

        keypoints3d_data = joints
        #{
        #    #camera_name: joints,
        #}

        metadata_data = {
            "metadata": {
                "camera_sync": sync_result,
                "source_frame_indices": {
                    camera_name: source_frame_id,
                },
            }
        }

        belief_output = {
            "camera": camera_name,
            "source_frame_id": source_frame_id,
            "pkl_frame_id": pkl_frame_id,
            "joints": belief_data,
        }

        # Giữ nguyên các file JSON
        write_json(
            keypoints_dir / f"pose_data_{out_frame_id}.json",
            keypoints3d_data,
        )

        write_json(
            metadata_dir / f"pose_data_{out_frame_id}.json",
            metadata_data,
        )

        write_json(
            belief_dir / f"belief_{out_frame_id}.json",
            belief_output,
        )

        # Tổng hợp dữ liệu cho một dòng CSV
        row = {
            "F_ID": out_frame_id,
            "Set": set_name,
            "Cam": camera_name,
            #"SourceF_ID": source_frame_id,
            "PKL_F_ID": pkl_frame_id,

            # Metadata: ánh xạ trực tiếp theo tên cột CSV
            "offset": sync_result["offset"],
            "left_start": sync_result["left_start"],
            f"PKLs": sync_result["source_pkls"][camera_name],
            f"FrameIndices": source_frame_id,
        }

        # Keypoints 3D:
        # Tạo camera1.head.x thay vì keypoints3d.camera1.head.x
        row.update(
            _flatten_dict(
                keypoints3d_data,
                prefix="",
            )
        )

        # Belief của từng joint
        row.update(
            flatten_belief_data(belief_data)
        )

        csv_rows.append(row)

    # Tạo tập hợp tất cả các cột xuất hiện trong dữ liệu
    fieldnames = [
        "F_ID",
        "Set",
        "Cam",
        "PKL_F_ID",
        "offset",
        "left_start",
        f"PKLs",
        f"FrameIndices",
    ]

    # Tọa độ 3D: giữ nguyên thứ tự joint trong YAML
    for joint_name in joint_names:
        for axis in ("x", "y", "z"):
            fieldnames.append(
                f"{joint_name}.{axis}"
            )

    # Belief: giữ nguyên thứ tự joint trong YAML
    for joint_name in joint_names:
        for attribute in (
            "global_belief",
            "local_belief",
            "visibility",
        ):
            fieldnames.append(
                f"{joint_name}.{MAPPING[attribute]}"
            )

    csv_path = output_dir / "pose_export.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[Pose] Done. Output: {output_dir}")
    print(f"[Pose] CSV exported: {csv_path}")
    print(f"[Pose] Frames exported: {len(csv_rows)}")
    print(f"[Pose] CSV columns: {len(fieldnames)}")

"""## 8. Fusion phase"""