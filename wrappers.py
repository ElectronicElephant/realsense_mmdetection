## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.
## Copyright(c) 2021 ElectronicElephant. All Rights Reserved.


import pyrealsense2 as rs
import numpy as np
import cv2


class AlignedCamera:
    def __init__(self, width=1280, height=720, fps=30) -> None:
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        print('Using', self.device_product_line, 'model...')
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def __del__(self):
        # Deconstructor
        self.pipeline.stop()

    def shot(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            raise RuntimeError('Failed to get frames.')

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image
    
    def vis(self, color_image, depth_image):
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        cv2.imshow('Align Example', images)

    def realtime_demo(self):
        # Streaming loop
        try:
            print('Starting realtime demo... Press q to quit.')
            while True:
                color_image, depth_image = self.shot()

                self.vis(color_image, depth_image)
                
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            self.pipeline.stop()


from mmdet.apis import init_detector, inference_detector, show_result_pyplot

class Detector:
    def __init__(self, config_file, checkpoint_file, device='cpu') -> None:
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device

        self.model = init_detector(config_file, checkpoint_file, device=device)

    def inference(self, img):
        # img: rgb or path to image
        return inference_detector(self.model, img)

    def vis(self, result, img, conf_threshold=0.5):
        result_img = self.model.show_result(img, result, score_thr=conf_threshold)
        cv2.imshow('Detection Result', result_img)

    def inference_and_vis(self, img, conf_threshold=0.5):
        result = self.inference(img)
        self.vis(result, img, conf_threshold)
        return result
