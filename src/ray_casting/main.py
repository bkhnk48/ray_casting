import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

#from models.keypoints_map import load_keypoints3d_map
from pose_pipeline.executor import load_pkl_data
from pose_pipeline.executor import run_pose_export

def main():
    pass
    #data = load_keypoints3d_map("nose")
    #print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl', type=str, 
                            default='inputs/datasets/mpi_inf_3dhpe/S8/Seq1/imageSequences/wham_video_0.pkl', 
                            help='input pkl file from WHAM')

    args = parser.parse_args()
    #data = load_keypoints3d_map(args.pkl)
    #lệnh dưới đây OK rồi
    #data = load_pkl_data(args.pkl)
    #data = 
    run_pose_export(args.pkl)
    #print(data)
    """print(type(data))

    if isinstance(data, dict):
        print(data.keys())
        for key, value in data.items():
            print(key, type(value), getattr(value, "shape", None))"""
    #main()