import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

#from models.keypoints_map import load_keypoints3d_map
from pose_pipeline.executor import load_pkl_data
from pose_pipeline.executor import run_pose_export
from preprocess_pipeline.annot_mat_to_csv import run_annot_mat_export

def main():
    pass
    #data = load_keypoints3d_map("nose")
    #print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl', type=str, 
                            default='inputs/datasets/mpi_inf_3dhpe/S8/Seq1/imageSequences/wham_video_0.pkl', 
                            help='input pkl file from WHAM')

    parser.add_argument('--ground_truth', type=str, 
                                default='inputs/datasets/mpi_inf_3dhpe/S8/Seq1/annot.mat', 
                                help='Ground Truth of the MPI_INF_3DHPE')

    parser.add_argument('--output', type=str, 
                                    default='outputs/datasets/mpi_inf_3dhpe/', 
                                    help='Output for an estimation of the MPI_INF_3DHPE')

    parser.add_argument('--smpl', type=str, 
                                        default="src/ray_casting/models/SMPL_NEUTRAL.pkl", 
                                        help='Default path for SMPL model')


    
    args = parser.parse_args()

    run_annot_mat_export(args)
    run_pose_export(args)
    """print(type(data))

    if isinstance(data, dict):
        print(data.keys())
        for key, value in data.items():
            print(key, type(value), getattr(value, "shape", None))"""
    #main()