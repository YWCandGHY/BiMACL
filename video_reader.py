import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import json
import pickle
from glob import glob

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


"""
Dataset for few-shot videos, which returns few-shot tasks. 
"""
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path  # 数据集地址
        self.seq_len = args.seq_len  # 8，一个视频几个帧表示
        self.train = True
        self.tensor_transform = transforms.ToTensor()  # 把读取的视频帧转变为tensor形式
        self.tensor_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tensor_clip_norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist  # 数据集few-shot划分地址

        self.way = args.way
        self.shot = args.shot
        self.query_per_class = args.query_per_class

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []
        # To tensor操作格式一般要求是PIL格式和numpy格式
        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        # 该操作是随机0.5概率水平翻转，只用于训练集, ssv2数据集理论不进行该数据增强
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
    
    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            # 因为压缩包压缩的就是二进制文件
            self.mem = open(zip_fn, 'rb').read()    # 压缩包以2进制文件打开，
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))  # BytesIO开辟IO缓存，多用于存储图片视频信息。StringIO存储字符信息
        else:       # self.zfile包含所有图片的压缩信息             # 不过和文件一样，使用完记得及时关闭回收内存空间。
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            # ssv2数据集中的图片格式原本就是jpg，提取出不包含.jpg的文件路径，也即一些目录文件之类的，因此称之为dir_list，剩下的就是图片文件
            # When using 'png' based datasets like kinetics, replace 'jpg' to 'png'
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))
            # x.split("/")) > 2目的是提取那些文件格式文件夹如下的：“ssv2/push/2048/”，差分后["ssv2","push","2048"," "],-3就是类别，所以觉class_folders
            # 但是有一个类是ssv2压缩包的名字，也要注意
            class_folders = list(set([x.split("/")[-3] for x in dir_list if len(x.split("/")) > 2]))
            class_folders.sort()
            self.class_folders = class_folders   # video_folders同上，-2是2048就是值得是视频的文件夹，ssv2理论上有1万个视频文件
            video_folders = list(set([x.split("/")[-2] for x in dir_list if len(x.split("/")) > 3]))
            video_folders.sort()
            self.video_folders = video_folders
            # 重新指定索引标号：这样每个视频文件夹和所有类别(甚至包含数据集压缩包名字)，压缩包名字小写，所以索引号100（第101个类别）
            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}
            # 该操作挑选出压缩包内所有.jpg文件地址
            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()  # 排列方式就按照了和class_folders的一样排列方式（因为图片文件夹路径，类别靠前）
            # video_folders[0] = “100001”号视频的编号， c默认其实就是的train_split,不过后面也没用到这值
            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:
                # [-3:0]，其值分别表示为类别文件夹，视频文件夹和视频每帧的jpg文件夹。
                class_folder, video_folder, jpg = img_path.split("/")[-3:]

                if video_folder != last_video_folder:
                    # 忽略小于seq.len的视频
                    if len(insert_frames) >= self.seq_len:
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]
               
                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)
        else:
            # 非zip文件处理方式，直接得出所有文件夹
            class_folders = os.listdir(self.data_dir)
            class_folders.sort()
            self.class_folders = class_folders
            for class_folder in class_folders:
                # 得到该类所有视频文件夹目录
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                # self.args.debug_loader作用是Load 1 vid per class for debugging
                if self.args.debug_loader:
                    video_folders = video_folders[0:1]
                for video_folder in video_folders:
                    # 返回视频属于测试还是训练的划分集
                    c = self.get_train_or_test_db(video_folder)
                    if c == None:
                        continue
                    # 根据目前类别视频文件夹得到所有该视频文件夹下的所有视频帧为imgs
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    # 忽略小于seq.len的视频
                    if len(imgs) < self.seq_len:
                        continue            
                    imgs.sort()  # 这里path也是列表，所以一个视频占split类的一个值
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id = class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used ，就是返回测试还是训练的划分集"""
    def get_train_or_test_db(self, split=None):
        if split is None:  # is None表示有值，说明读取的压缩包有视频文件夹
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                # train list种data的长度为6400，每类100个视频，然后就是训练集64类，所以是64个视频，list中元素类别为str
                data = fid.readlines()
                # str.replace(old, new，[, max])，第一个元素是被替换元素，第二个是替换元素，第三个是替换的最大次数
                # 通过此把类别中的空格换成了"_"
                data = [x.replace(' ', '_').lower() for x in data]
                # strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列,注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
                # 对应的结果就是读取的每行的视频信息最后的“\n”消失，split(" ")消除空格
                data = [x.strip().split(" ")[0] for x in data]
                # os.path.splitext(“文件路径”) ,分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作,本例其实没有扩展名
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                # os.path.split(x)[1]表示取后面的视频指标数字
#                 if "kinetics" in self.args.path:
#                     data = [x[0:11] for x in data]
                # extend表示往列表一次添加多个元素
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i
    
    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx) 
        n_frames = len(paths)
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2
    
            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            
            imgs = [self.tensor_norm(self.tensor_transform(v)) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id

    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        # select classes to use for this task  get_train_split = self.train，因为测试时候self.train=FLASE，所以c对应为test.split()
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        # # 用以生成索引标签，仅第一次训练时候使用下就行了
        # cla_dict = {}
        # for i in range(len(classes)):
        #     cla_dict[str(classes[i])] = i
        # # write dict into json file
        # json_str = json.dumps(cla_dict, indent=4)
        # with open('class_indices_ucf101_train.json', 'w') as json_file:
        #     json_file.write(json_str)
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
            for idx in idxs[self.args.shot:]:
                vid, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
        
        s = list(zip(support_set, support_labels))
        random.shuffle(s)
        support_set, support_labels = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes}


