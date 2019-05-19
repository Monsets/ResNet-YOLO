class Config:

    def __init__(self):

        self.batch_size = 32  # orig paper trained all networks with batch_size=128
        self.epochs = 200
        self.data_augmentation = True
        self.num_classes = 2
        # Subtracting pixel mean improves accuracy
        self.subtract_pixel_mean = True
        # Shuffle data after epoch end
        self.shuffle = True

        # Model parameter
        # ----------------------------------------------------------------------------
        #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
        # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
        #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
        # ----------------------------------------------------------------------------
        # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
        # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
        # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
        # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
        # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
        # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
        # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
        # ---------------------------------------------------------------------------
        self.n = 6

        # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
        self.version = 2

        # Computed depth from supplied model parameter n
        if self.version == 1:
            self.depth = self.n * 6 + 2
        elif self.version == 2:
            self.depth = self.n * 9 + 2

        # Model name, depth and version
        self.model_type = 'ResNet%dv%d' % (self.depth, self.version)
        # number of zones to split image
        self.num_areas = 15
        # number of boxes for each area to predict
        self.B = 1
        # path to labels
        self.labels_path = 'data/attributes.txt'
        # test size
        self.test_size = 0.1
        # size to reshape for network
        self.resize_shape = 240
        # if image box area % lies within bounding box more than area_treshold then accept it as a label
        self.area_threshold = 0.5


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
