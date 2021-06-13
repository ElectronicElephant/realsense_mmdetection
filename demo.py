## License: Apache 2.0. See LICENSE file in root directory.
import numpy as np
import cv2
from torch.nn.functional import threshold

from wrappers import AlignedCamera, Detector
from tqdm import tqdm


# ============================================ #
### RealSense Camera
# ============================================ #

cam = AlignedCamera(1280, 720, 30)
# cam.realtime_demo()


# ============================================ #
### MMDetection
# ============================================ #

config_file = 'models/fcos_coco_randbg8000_0.1_0.5.py'
checkpoint_file = 'models/fcos_coco_randbg8000_0.1_0.5.pth'
device = 'cuda:0'
det = Detector(config_file, checkpoint_file, device) 


# ============================================ #
### Main Loop
# ============================================ #

pbar = tqdm(total=0, ncols=0)
while True:
    bgr, depth = cam.shot()
    cam.vis(bgr, depth)

    result = det.inference_and_vis(bgr, conf_threshold=0.15)
    # print(result)

    # ====== post process ====== #
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
    pbar.update(1)

pbar.close()
