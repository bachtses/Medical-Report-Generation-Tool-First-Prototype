class Dataset(torch.utils.data.Dataset):
  def __init__(self, path_to_imgs, set_dict):
    self.path_to_imgs = path_to_imgs
    self.imgs_dict = set_dict

  def __len__(self):
    return len(self.imgs_dict)

  def __getitem__(self, index):

    img_info = self.imgs_dict[index]
    img_path = self.path_to_imgs+img_info["filename"]+'.npy'
    temp     = np.load(img_path)
    b        = img_info["label"]
    temp     = (temp - np.min(temp))/(np.max(temp) - np.min(temp))


    if (b == "1"):
      y = np.array([1, 0, 0, 0], dtype=np.float16)
    if (b == "2"):
      y = np.array([0, 1, 0, 0], dtype=np.float16)
    if (b == "3"):
      y = np.array([0, 0, 1, 0], dtype=np.float16)
    if (b == "4"):
      y = np.array([0, 0, 0, 1], dtype=np.float16)

    temp = np.array(temp, dtype=np.float32)
    x    = torch.from_numpy(temp)

    return x, y
