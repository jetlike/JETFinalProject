import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision.hand_tracker import get_pointing_target

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.br = CvBridge()
        self.sub = self.create_subscription(
            String, 'llm_query', self.on_query, 10
        )
        self.pub = self.create_publisher(Image, 'cropped_image', 10)

    def on_query(self, msg: String):
        label, path = get_pointing_target(cooldown_secs=1.0)
        if path:
            img = cv2.imread(path)
            ros_img = self.br.cv2_to_imgmsg(img, encoding='bgr8')
            self.pub.publish(ros_img)
            self.get_logger().info(f'Published crop: {label}')

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
