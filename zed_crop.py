import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class ZedCropNode(Node):
    def __init__(self):
        super().__init__('zed_crop_node')

        self.bridge = CvBridge()

        self.img_subscription = self.create_subscription(
            Image, 
            '/image_raw',
            self.listener_callback,
            10
        )
        self.info_subscription = self.create_subscription(
            CameraInfo, 
            '/camera_info',
            self.info_callback,
            10
        )

        self.img_left_publisher = self.create_publisher(CompressedImage, '/zed/zed_node/left/color/raw/image/compressed', 10)
        self.info_left_publisher = self.create_publisher(CameraInfo, '/zed/zed_node/left/color/raw/camera_info', 10)

        self.img_right_publisher = self.create_publisher(CompressedImage, '/zed/zed_node/right/color/raw/image/compressed', 10)
        self.info_right_publisher = self.create_publisher(CameraInfo, '/zed/zed_node/right/color/raw/camera_info', 10)

        self.thrid_person_publisher = self.create_publisher(Image, '/rs/third_person_cropped', 10 )

        self.latest_info = None

        self.get_logger().info('ZED crop image active')

    def info_callback(self, msg):
        new_info = msg
        new_info.width = msg.width // 2

        self.latest_info = new_info


    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
 
        height, width, _ = cv_image.shape
        half_width = width // 2
        right_img = cv_image[0:height, 0:half_width]
        left_img = cv_image[0:height, half_width:width]

        left_img = cv2.flip(left_img, 0)
        right_img = cv2.flip(right_img, 0)

        left_img = cv2.flip(left_img, 1)
        right_img = cv2.flip(right_img, 1)
    
        msg = self._to_compressed_msg(right_img, data.header)
        self.img_right_publisher.publish(msg)
 
        if self.latest_info:
            self.latest_info.header = data.header
            self.info_right_publisher.publish(self.latest_info)
 
        msg2 = self._to_compressed_msg(left_img, data.header)
        self.img_left_publisher.publish(msg2)
 
        if self.latest_info:
            self.latest_info.header = data.header
            self.info_left_publisher.publish(self.latest_info)
 
    def _to_compressed_msg(self, cv_image, header):
        success, encoded = cv2.imencode('.jpg', cv_image)
        msg = CompressedImage()
        msg.header = header
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        return msg

def main(args=None):

    rclpy.init(args=args)
    node = ZedCropNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()