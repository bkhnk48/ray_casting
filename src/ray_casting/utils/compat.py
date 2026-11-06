
import sys
import numpy as np
import joblib
import inspect
from pathlib import Path
import pdb
import pandas as pd

from utils.constants import *

def patch_numpy_aliases():
    aliases = {
        "float": float,
        "int": int,
        "bool": bool,
        "object": object,
        "typeDict": np.sctypeDict,
        "complex": complex,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def patch_inspect():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def patch_numpy_and_inspect():
    patch_numpy_aliases()
    patch_inspect()


def configure_stdout_encoding():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_joblib_compat(filepath):
    with open(filepath, "rb") as file:
        return joblib.load(file)


def find_person_mapping_files(pkl_path):
    if pkl_path is None:
        return []

    parent = Path(pkl_path).parent
    candidates = list(parent.glob("*person_mapping.txt"))

    if not candidates and parent.parent.exists():
        candidates = list(
            parent.parent.glob("**/*person_mapping.txt")
        )

    return candidates


def read_person_mapping(mapping_file):
    mapping = {}

    for line in mapping_file.read_text(
        encoding="utf-8"
    ).splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        mapping[key.strip()] = value.strip().lower()

    return mapping


def parse_person_mapping(pkl_path):
    candidates = find_person_mapping_files(pkl_path)

    if not candidates:
        return {}

    return read_person_mapping(candidates[0])


def is_person_dict(value):
    return isinstance(value, dict)


def get_pose_length(person_data):
    if not is_person_dict(person_data):
        return 0

    return len(person_data.get("pose", []))


def resolve_track_key(track_str, wham_data):
    track_id = int(track_str) if track_str.isdigit() else track_str

    if track_id in wham_data:
        return track_id

    if str(track_id) in wham_data:
        return str(track_id)

    return None


def find_tracks_by_person(wham_data, mapping, target_str):
    candidates = {}


    if target_str == ALL.lower():
        for track_id in wham_data.keys():
            candidates[track_id] = wham_data[track_id]
        return candidates


    for track_str, person_name in mapping.items():
        if person_name != target_str:
            continue

        track_id = resolve_track_key(track_str, wham_data)

        if track_id is not None:
            candidates[track_id] = wham_data[track_id]
    return candidates


def select_longest_person(candidates):
    if not candidates:
        return None
    print("Có khả năng lỗi ở đây, cẩn thận khi candidates từ 17h10, 6/11/26 là dict thay vì list")
    return max(candidates, key=get_pose_length)


def resolve_mapped_person(wham_data, pkl_path, target_str):
    mapping = parse_person_mapping(pkl_path)
    candidates = find_tracks_by_person(
        wham_data, mapping, target_str
    )

    if candidates:
        return select_longest_person(candidates)

    if mapping:
        return None

    return None


def find_person_by_target_key(wham_data, target_person, target_str):
    if target_person in wham_data:
        return wham_data[target_person]

    if target_str in wham_data:
        return wham_data[target_str]

    if target_str.isdigit():
        numeric_id = int(target_str)
        if numeric_id in wham_data:
            return wham_data[numeric_id]

    return None


def resolve_target_person(wham_data, pkl_path, target_person):
    target_str = str(target_person).strip().lower()

    if pkl_path is not None:
        mapping = parse_person_mapping(pkl_path)
        candidates = find_tracks_by_person(
            wham_data, mapping, target_str
        )
        if target_person == ALL:
            return candidates

        if candidates:
            return select_longest_person(candidates)

        # Strict identity: không thay thế bằng người khác.
        if mapping:
            return None

    return find_person_by_target_key(
        wham_data, target_person, target_str
    )


def find_default_person(wham_data):
    if 0 in wham_data:
        return wham_data[0]

    if "0" in wham_data:
        return wham_data["0"]

    candidates = [
        value
        for value in wham_data.values()
        if is_person_dict(value) and "pose" in value
    ]

    if candidates:
        return max(candidates, key=get_pose_length)

    for value in wham_data.values():
        if is_person_dict(value):
            return value

    return None


def resolve_non_mapping_payload(wham_data):
    if isinstance(wham_data, list):
        for value in wham_data:
            if is_person_dict(value):
                return value

    return None


def is_mapping_payload(wham_data):
    return (
        isinstance(wham_data, dict)
        or "defaultdict" in str(type(wham_data))
    )


def resolve_person_payload(wham_data, pkl_path=None, target_person=ALL):
    if not is_mapping_payload(wham_data):
        return resolve_non_mapping_payload(wham_data)

    if target_person is not None:
        return resolve_target_person(
            wham_data, pkl_path, target_person
        )

    return find_default_person(wham_data)
