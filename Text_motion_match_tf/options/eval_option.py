
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
        default="./checkpoints_grab/model")
    parser.add_argument("--data_root", type=str,
        default="./data/saved_motion_pairs")
    parser.add_argument("--log_file", type=str,
        default="./grab_ablavion.log")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--replication_times", type=int, default=20)
    parser.add_argument("--diversity_times", type=int, default=70)
    parser.add_argument("--mm_num_times", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser

def get_args():
    return get_parser().parse_args()
