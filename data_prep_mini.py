import os
import random
from collections import defaultdict

dataset_root = '/scratch/datasets/something-something_v2/Frames'
list_dir = '/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/'

def split_file(file, unlabeled, labeled, percentage, isShuffle=True, seed=123, strategy='classwise'):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    if strategy == 'classwise':
        # if os.path.exists(unlabeled) and os.path.exists(labeled):
        #   print("path exists with this seed and strategy")
        #   return 
        random.seed(seed)
        #creating dictionary against each category
        def del_list(list_delete,indices_to_delete):
            for i in sorted(indices_to_delete, reverse=True):
                del(list_delete[i])

        main_dict= defaultdict(list)
        with open(os.path.join(list_dir, file),'r') as mainfile:
            lines = mainfile.readlines()
            print(f"Total videos: {len(lines)}")
            for line in lines:
                video_info = line.strip().split()
                main_dict[video_info[3]].append((video_info[0],video_info[1], video_info[2]))
        with open(os.path.join(list_dir, unlabeled),'w') as ul,\
            open(os.path.join(list_dir, labeled),'w') as l:
            for key,value in main_dict.items():
                length_videos = len(value)
                ul_no_videos = int((length_videos* percentage))
                indices = random.sample(range(length_videos),ul_no_videos)
                for index in indices:
                    line_to_written = value[index][0] + " " + value[index][1] + " " + value[index][2] + " " +key+"\n"
                    ul.write(line_to_written)
                del_list(value,indices)
                for label_index in range(len(value)):
                    line_to_written = value[label_index][0] + " " + value[label_index][1] + " " + value[label_index][2] + " " +key+"\n"
                    l.write(line_to_written)
        

        


    if strategy == 'overall':
        if os.path.exists(unlabeled) and os.path.exists(labeled):
          print("path exists with this seed and strategy")
          return 
        random.seed(seed)
        with open(file, 'r') as fin, \
            open(unlabeled, 'w') as foutBig, \
            open(labeled, 'w') as foutSmall:
        # if didn't count you could only approximate the percentage
            lines = fin.readlines()
            random.shuffle(lines)
            nLines = sum(1 for line in lines)
            nTrain = int(nLines*percentage)
            i = 0
            for line in lines:
                line = line.rstrip('\n') + "\n"
                if i < nTrain:
                     foutBig.write(line)
                     i += 1
                else:
                     foutSmall.write(line)




def create_file(path, outfile):
    with open(path, 'r') as fin, open(os.path.join(list_dir + outfile), 'w') as fout:
        lines = fin.readlines()
        for line in lines:
            vid, frames, label = line.split(' ')
            frame_dir = os.path.join(dataset_root, vid)
            if os.path.exists(frame_dir):
                record = [frame_dir, str(1), str(frames), str(label)]
                record = [r.strip() for r in record]
                print(" ".join(record), file=fout)



def main():
   
    
    trainlist_path = '/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/mini_train_mmaction2.txt'
    testlist_path = '/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/mini_val_mmaction2.txt'
    
    # create_file(trainlist_path, "mini_train.txt")
    # create_file(testlist_path, "mini_test.txt")

    split_file("mini_train.txt", "unlabeled_training_minisv2.txt", "labeled_training_minisv2.txt", 0.95)



if __name__ == "__main__":
    main()
