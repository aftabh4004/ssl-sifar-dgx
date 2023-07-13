from sifar_pytorch.video_dataset import VideoDataSet, VideoDataSetLMDB, VideoDataSetOnline

import torch
try:
    import lmdb
    import pyarrow as pa
    _HAS_LMDB = True
except ImportError as e:
    _HAS_LMDB = False
    _LMDB_ERROR_MSG = e

try:
    import av
    _HAS_PYAV = True
except ImportError as e:
    _HAS_PYAV = False
    _PYAV_ERROR_MSG = e

class VideoDataSetOnlineSSL(VideoDataSetOnline):
    def __init__(self, root_path, list_file, num_groups=8, frames_per_group=1, sample_offset=0,
                 num_clips=1, modality='rgb', dense_sampling=False, fixed_offset=True,
                 image_tmpl='{:05d}.jpg', transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=None, whole_video=False,
                 fps=29.97, audio_length=1.28, resampling_rate=24000, isLabeled=True):
        

        if not _HAS_PYAV:
            raise ValueError(_PYAV_ERROR_MSG)
        if modality not in ['rgb', 'rgbdiff']:
            raise ValueError("modality should be 'rgb' or 'rgbdiff'.")

        super().__init__(root_path, list_file, num_groups, frames_per_group, sample_offset,
                         num_clips, modality, dense_sampling, fixed_offset,
                         image_tmpl, transform, is_train, test_mode, seperator,
                         filter_video, num_classes, whole_video, fps, audio_length, resampling_rate)
        
        self.isLabeled = isLabeled
    
    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """
        record = self.video_list[index]
        # check this is a legit video folder
        indices = self._sample_indices(record) if self.is_train else self._get_val_indices(record)
        images = self.get_data(record, indices)
        images = self.transform(images)
        label = self.get_label(record)

        # # for unlabeled dataloading
        # if not self.isLabeled:
        #     seg4_image = [images[i:i+3, :, : ] for i in range(0, 24, 6)]
        #     seg4_image = torch.cat(seg4_image, dim=0)
        #     return images, seg4_image

        # re-order data to targeted format.
        return images, label