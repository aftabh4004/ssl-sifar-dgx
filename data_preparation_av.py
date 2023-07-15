import cv2
from ssl_sifar_utils import get_training_filenames, validate_split

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count



def create_label_dict(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()
        label_to_idx = {}
        for line in lines:
            index, label = line.split()
            label_to_idx[label.strip()] = int(index.strip()) - 1
        
    return label_to_idx

def create_train_val_list(dataset_root, listpath, label_to_idx, list_dir):
    
    with open(listpath, 'r') as fp:
        lines = fp.readlines()
        train_list = []
        count = 0
        
        for line in lines:
            vpath = line.split()[0]
            label = vpath.split('/')[0]
            vpath = dataset_root + vpath
            fcount = get_video_frame_count(vpath)
            train_list += [[vpath, str(1), str(fcount), str(label_to_idx[label])]]
            if count % 50 == 0:
                print(f"done {count}")
            count += 1
        

       
        with open(list_dir + 'train.txt', 'w') as fp:
            for record in train_list:
                print(" ".join(record), file=fp)
            print("train.txt saved")
        return list_dir + 'train.txt'
       
            

def create_test_list(dataset_root, listpath, label_to_idx, list_dir):
     with open(listpath, 'r') as fp:
        lines = fp.readlines()
        test_list = []
        count = 0

        for line in lines:
            vpath = line.split()[0]
            label = vpath.split('/')[0]
            vpath = dataset_root + vpath
            fcount = get_video_frame_count(vpath)
            test_list += [[vpath, str(1), str(fcount), str(label_to_idx[label])]]
            if count % 50 == 0:
                print(f"done {count}")
            count += 1

        with open(list_dir + 'test.txt', 'w') as fp:
            for record in test_list:
                print(" ".join(record), file=fp)
            print("test.txt saved")

def main():
    dataset_root = '/scratch/datasets/UCF-101/Videos/'
    list_dir = '/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/'
    classind_path = "/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/classInd.txt"
    trainlist_path = "/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/ucf101_train_split_1_videos.txt"
    testlist_path = "/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/ucf101_val_split_1_videos.txt"

    label_to_idx = create_label_dict(classind_path)
    print(label_to_idx)
    train_list = create_train_val_list(dataset_root, trainlist_path, label_to_idx, list_dir)
    create_test_list(dataset_root, testlist_path, label_to_idx, list_dir)

    train_label_list, train_unlabel_list = get_training_filenames(list_dir, train_list, 0.95, 'classwise')

    validate_split(train_label_list, train_unlabel_list)



if __name__ == "__main__":
    main()
