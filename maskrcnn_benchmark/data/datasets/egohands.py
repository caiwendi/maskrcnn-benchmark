from PIL import Image
import os
import scipy.io as scio
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList

activities = ['CARDS', 'CHESS', 'JENGA', 'PUZZLE']
locations = ['_COURTYARD', '_LIVINGROOM', '_OFFICE']
actors = ['_B', '_T', '_H', '_S']


class EgohandsDataset(object):
    def __init__(self, act_idx, loc_idx, actor1_idx, actor2_idx, transforms=None):
        matfiles = []
        for a in act_idx:
            act = activities[a]
            for l in loc_idx:
                act_loc = act + locations[l]
                for m in actor1_idx:
                    act_loc_actor_1 = act_loc + actors[m]
                    for n in actor2_idx:
                        matfiles.append(os.path.join(
                            '/home/lab/PycharmProjects/Hand-Detection/datasets/egohands/labels',
                            act_loc_actor_1 + actors[n] + '.mat'))
        self.image_dir = []
        self.bboxes = []
        for f in matfiles:
            if os.path.exists(f):
                matdata = scio.loadmat(f)
                self.image_dir.extend(matdata["images_name"])
                self.bboxes.extend(matdata["boxlist"][0])
        self.transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(self.image_dir[idx])
        # self.height, self.width = image.size
        bbox = self.bboxes[idx]
        if bbox.shape[0] != 0:
            box = bbox[:, :-1]
            label = bbox[:, -1]
        else:
            box = [[0.0, 0.0, 0.0, 0.0]]
            label = [0]
        boxlist = BoxList(box, image.size, mode="xyxy")
        classes = torch.tensor(label)
        boxlist.add_field("labels", classes)
        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)
        return image, boxlist, idx

    def __len__(self):
        return len(self.image_dir)

    def get_img_info(self, idx):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        return {"width": 1280, "height": 720}

