import os.path as osp

import archs  # noqa: F401
import data  # noqa: F401
import models  # noqa: F401
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
