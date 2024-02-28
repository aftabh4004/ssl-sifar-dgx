import argparse
from pathlib import Path
import os
import av

def get_video_frame_count(video_path):
    try:
        container = av.open(video_path)
    except:
        print(f"Error in {video_path}")
        return None

    nframes = container.streams.video[0].frames
    return nframes

def get_args_parser():
    parser = argparse.ArgumentParser('Datapreparation script', add_help=False)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--output_dir', default='/dataset_list/default', type=str)
    parser.add_argument('--input_dir',  type=str)
    
    return parser

def main(args):
    files = os.listdir(args.input_dir)
    for file in files:
        outfile = ""
        if file == 'val.csv':
            continue
        elif file == 'test.csv':
            outfile = 'test.txt'
        elif file == 'train.csv':
            outfile = 'labeled_training.txt'
        elif file == 'unlabel.csv':
            outfile = 'unlabeled_training.txt'
            
        print(f"Processing {file}")
    
        with open(os.path.join(args.input_dir, file), 'r') as fp, open(os.path.join(args.output_dir, outfile), 'w') as fpo:
            lines = fp.readlines()
            n = len(lines)
            c = 0
            for line in lines:
                path, lable = line.strip().split()
                path = "/".join(path.split('/')[-2:])
                nframes = get_video_frame_count(os.path.join(args.dataset_root, path))
                if nframes is None:
                    continue
                print(f"{path} 1 {nframes} {lable}", file=fpo)
                c += 1
                if c % 100 == 0:
                    print(f"[{c}/{n}]")
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Dataset perperation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# python3 process_SVformer_list.py --output_dir "./dataset_list/ucf101_10per_SVformer" --input_dir "/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/dataset_list_SVformer/list_ucf_10" --dataset_root "/home/mt0/22CS60R54/datasets/ucf101/videos/"

# /home/mt0/22CS60R54/datasets/hmdb51/videos

# python3 process_SVformer_list.py --output_dir "./dataset_list/hmdb51_40per_SVformer" --input_dir "/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/dataset_list_SVformer/list_hmdb_40" --dataset_root "/home/mt0/22CS60R54/datasets/hmdb51/"
