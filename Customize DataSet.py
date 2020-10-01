from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, label_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dir = label_dir
        self.root_dir = root_dir
        self.path = os.path.join(root_dir, label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, item):
        image_name = self.image_path[item]
        img_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return label, img

    def __len__(self):
        return len(self.image_path)


root = "D:\ANU\Deep Learning CV\Cat-Dog-data\Cat-Dog-data\cat-dog-train"
cat = "cats"
catSet = MyData(root, cat)

root = "D:\ANU\Deep Learning CV\Cat-Dog-data\Cat-Dog-data\cat-dog-train"
dog = "dogs"
dogSet = MyData(root, dog)
