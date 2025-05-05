import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from llm.query_engine import QueryEngine

class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.br = CvBridge()
        self.engine = QueryEngine(model='gpt-4o-inference', temperature=0.3)
        self.query = None
        self.img   = None

        self.qsub = self.create_subscription(
            String, 'llm_query', self.q_cb, 10
        )
        self.isub = self.create_subscription(
            Image, 'cropped_image', self.i_cb, 10
        )
        self.pub = self.create_publisher(String, 'llm_response', 10)

    def q_cb(self, msg: String):
        self.query = msg.data
        self.try_respond()

    def i_cb(self, msg: Image):
        self.img = msg
        self.try_respond()

    def try_respond(self):
        if self.query and self.img:
            answer = self.engine.answer(
                question=self.query,
                context_text='[image attached]'
            )
            out = String(data=answer)
            self.pub.publish(out)
            self.get_logger().info('Published LLM answer')
            self.query = None
            self.img   = None

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
