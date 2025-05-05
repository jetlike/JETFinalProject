import os, threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from audio.listener import WakeWordListener

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener_node')
        self.pub = self.create_publisher(String, 'llm_query', 10)
        self.face_sub = self.create_subscription(
            Bool, 'face_verified', self.face_cb, 10
        )
        self.listener = None
        self.listening = False

    def face_cb(self, msg: Bool):
        if msg.data and not self.listening:
            self.get_logger().info('Face verified → starting listener')
            self.start_listener()
        elif not msg.data and self.listening:
            self.get_logger().info('Face lost → stopping listener')
            self.stop_listener()

    def start_listener(self):
        keyword_paths = [os.getenv('PROJECT_ROOT') + '/models/hey_bot.ppn']
        self.listener = WakeWordListener(
            keyword_paths=keyword_paths,
            sensitivities=[0.6],
            callback=self.on_wake
        )
        threading.Thread(target=self.listener.start, daemon=True).start()
        self.listening = True

    def stop_listener(self):
        if self.listener:
            self.listener.stop()
        self.listening = False

    def on_wake(self):
        wav = self.listener._record_with_threshold()
        transcript = self.listener.transcriber.transcribe(wav)
        msg = String(data=transcript)
        self.pub.publish(msg)
        self.get_logger().info(f'Published transcript: "{transcript}"')

def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()
    rclpy.spin(node)
    node.stop_listener()
    node.destroy_node()
    rclpy.shutdown()
