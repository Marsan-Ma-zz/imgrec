import tflearn.datasets.oxflower17 as oxflower17
oxflower17.load_data(dirname='images/17flowers', one_hot=True, resize_pics=(227, 227))
