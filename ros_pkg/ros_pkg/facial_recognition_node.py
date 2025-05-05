import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import cv2
from face.facial_recog import FaceRecognitionSystem

class FacialRecognitionNode(Node):
    def __init__(self):
        super().__init__('facial_recognition_node')
        self.pub = self.create_publisher(Bool, 'face_verified', 10)
        self.face_sys = FaceRecognitionSystem()
        self.cap = cv2.VideoCapture(0)
        self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        name = self.face_sys.recognize_once(frame)
        verified = name not in ("Unknown", "NoFace")
        msg = Bool(data=verified)
        self.pub.publish(msg)
        self.get_logger().debug(f"FaceRecognition: {name} → {verified}")

def main(args=None):
    rclpy.init(args=args)
    node = FacialRecognitionNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()
