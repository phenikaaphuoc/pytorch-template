import random
from src.utils import logger
import cv2
import numpy as np
import os
from src.utils.utils import *
import argparse
_VIDEO_EXT = ['.avi', '.mp4', '.mov']
_IMAGE_EXT = ['.jpg', '.png']
_IMAGE_SIZE = 224


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n))]


def resize_img(img, short_size=256):
    h, w, c = img.shape
    if (w <= h and w == short_size) or (h <= w and h == short_size):
        return img
    if w < h:
        ow = short_size
        oh = int(short_size * h / w)
        return cv2.resize(img, (ow, oh))
    else:
        oh = short_size
        ow = int(short_size * w / h)
        return cv2.resize(img, (ow, oh))


def video_loader(video_path, short_size):
    video = []
    vidcap = cv2.VideoCapture(str(video_path))
    major_ver, *_ = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    video.append(resize_img(image, short_size))
    while success:
        success, image = vidcap.read()
        if not success: break
        video.append(resize_img(image, short_size))
    vidcap.release()

    return video, len(video), fps


def images_loader(images_path, transform=None):
    images_set = []
    images_list = [i for i in images_path.iterdir() if not i.stem.startswith('.') and i.suffix.lower() in _IMAGE_EXT]
    images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f.stem))))
    for image_path in images_list:
        image = cv2.imread(str(image_path), 3)
        if transform is not None:
            image = transform(image)
        images_set.append(image)
    return images_set, len(images_set)


def sample_by_number(frame_num, out_frame_num, random_choice=False):
    full_frame_lists = split(list(range(frame_num)), out_frame_num)

    if random_choice:
            return [random.choice(i) for i in full_frame_lists]
    else:
        return [i[0] for i in full_frame_lists]




class FrameGenerator(object):
    def __init__(self, input_path,
                 sample_num=1,
                 resize=None,
                 ):
        """
        :param input_path: The input video file or image set path
        :param sample_num: The number of frames you hope to use, they are chosen evenly spaced
        :param slice_num: The number of blocks you want to divide the input file into, and frames
                            are randomly chosen from each block.
        """
        input_path = Path(input_path)
        self.is_video = input_path.is_file() and input_path.suffix.lower() in _VIDEO_EXT
        if self.is_video:
            self.frames, self.frame_num, self.fps = video_loader(input_path, resize)
        elif input_path.is_dir():
            self.frames, self.frame_num = images_loader(input_path, resize)
        else:
            raise IOError("Input data path is not valid! Please make sure it is whether "
                          "a video file or a image set directory")
        self.counter = 0
        self.current_video_frame = -1
        self.sample_num = sample_num
        self.chosen_frames = sample_by_number(self.frame_num, sample_num, False)

    def __len__(self):
        return len(self.chosen_frames)

    def reset(self):
        self.counter = 0

    def get_frame(self):
        frame = self.frames[self.counter]  # cv2.resize(frame, (_IMAGE_SIZE, _IMAGE_SIZE))
        self.counter += 1
        return frame


def get_video_generator(video_path, opts):
    
    out_path = Path(opts.out_path,
                        video_path.parts[-2])
    os.makedirs(out_path,exist_ok=True)
    if not out_path.exists(): out_path.mkdir()
    out_path_dic = {}
    out_path_dic["rgb"] = out_path / ('{}.npy'.format(video_path.stem))
    video_object = FrameGenerator(video_path,
                                  opts.sample_num,
                                  resize=opts.resize, 
                                )
    return video_object, out_path_dic


def compute_rgb(video_object, out_path):
    """Compute RGB"""
    
    rgb = np.array(video_object.frames)[:,:]
    np.save(out_path["rgb"], rgb)
    return rgb

def pre_process(video_path, opts):
    video_path = Path(video_path)
    video_object, out_path_dic = get_video_generator(video_path, opts)
    rgb_data = compute_rgb(video_object, out_path_dic)
    video_object.reset()
    return rgb_data


def mass_process(opts):
    out_root_folder = opts.out_path
   
    if type(opts.data_root) is list:
        from tqdm import tqdm
        for data_root in tqdm(opts.data_root):
            data_root = Path(data_root)
            class_paths = [i for i in data_root.iterdir() if not i.stem.startswith(".") and i.is_dir()]
            item_paths = []
            for class_path in class_paths:
                    item_paths.extend([i for i in class_path.iterdir()
                                    if not i.stem.startswith(".") and i.is_file() and i.suffix.lower() in _VIDEO_EXT])
            input_folder_name = os.path.basename(data_root)
            opts.out_path =os.path.join(out_root_folder,input_folder_name)
          
            os.makedirs(opts.out_path,exist_ok=True)
            for item_path in item_paths:
                with Timer(item_path.name):
                    pre_process(item_path, opts)
    else:
        data_root = Path(opts.data_root)
        class_paths = [i for i in data_root.iterdir() if not i.stem.startswith(".") and i.is_dir()]
        item_paths = []
        for class_path in class_paths:
                item_paths.extend([i for i in class_path.iterdir()
                                if not i.stem.startswith(".") and i.is_file() and i.suffix.lower() in _VIDEO_EXT])
        for item_path in item_paths:
            with Timer(item_path.name):
                pre_process(item_path, opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-process the video into formats which i3d uses.')
    parser.add_argument(
        '--data_root',
        type=str,
        default=[r"D:\phuoc_sign\dataset\cross_data_split_by_person\1",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\2",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\3",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\4",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\5",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\6",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\7",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\8",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\9",
                 r"D:\phuoc_sign\dataset\cross_data_split_by_person\10",],
        help='Where you want to save the output input_folder')
    parser.add_argument(
        '--out_path',
        type=str,
        default=r"D:\phuoc_sign\dataset\raw_data_set_cross_split",
        help='Where you want to save the output rgb')
    # Sample arguments
    parser.add_argument(
        '--sample_num',
        type=int,
        default='32',
        help='The number of the output frames after the sample, or 1/sample_rate frames will be chosen.')
    parser.add_argument(
        '--resize',
        type=int,
        default='256',
        help="Resize the short edge video to '--resize'. Mention that this is only the pre-process, random crop"
             "will be applied later when training or testing, so here 'resize' can be a little bigger.")
    parser.add_argument(
        '--random_choice',
        action='store_true',
        help='Whether to choose frames randomly or uniformly')
    parser.add_argument(
        '--sample_type',
        type=str,
        default='num',
        help="'fps': sample the video to a certain FPS, or 'num': control the number of output video, "
             "choose the video sample method.")

    args = parser.parse_args()
    os.makedirs(args.out_path,exist_ok=True)
    mass_process(args)

