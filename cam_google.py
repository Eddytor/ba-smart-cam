"""cam google

This module send the image information required to run through the ml model at a google cloud engine vm.
It will connect to the MQTT bridge from the IoT Core.
"""
import time
import numpy as np
import cv2
import paho.mqtt.client as mqtt
from absl import app, flags
from paho.mqtt.client import ssl

import utility

FLAGS = flags.FLAGS
flags.DEFINE_string('project_id', 'smart-cam-ba', 'project id of google cloud platform')
flags.DEFINE_string('cloud_region', 'europe-west1', 'cloud region of the google IoT Core registry')
flags.DEFINE_string('registry_id', 'cam', 'registry id in the google IoT Core')
flags.DEFINE_string('device_id', 'cam-google', 'device id that is already added in the corresponding registry id')
flags.DEFINE_string('algorithm', 'RS256', 'encryption algorithm to use for jwt')
flags.DEFINE_string('private_key_file', './data/rsa_private.pem', 'file path to the private key .pem')
flags.DEFINE_string('google_mqtt_server_ca', './data/roots.pem',
                    'file path to the google root CA certification package')
flags.DEFINE_string('mqtt_bridge_hostname', 'mqtt.googleapis.com', 'MQTT bridge hostname')
flags.DEFINE_integer('mqtt_bridge_port', 8883, 'MQTT bridge port')
flags.DEFINE_integer('input_size', 416, 'size of img in height/width')
flags.DEFINE_integer('interval_time', 1, 'time in which the cam sends images to the broker')
flags.DEFINE_string('cam_type', 'webcam', 'type of the video stream')
flags.DEFINE_string('cam_link', None,
                    'link to the video e.g. ip addr for ip_cam or path to video, for webcam not needed')
flags.DEFINE_boolean('show_stream', True, 'display the stream of camera in window')


class Cam:
    def __init__(self):
        self.uptime = utility.get_datetime()
        self.client_id = None
        self.topic = None
        self.client = self.init_cam()

        self.client.will_set(f"/devices/{FLAGS.device_id}/state", 1, qos=1, retain=True)
        self.client.loop_start()

    @staticmethod
    def on_disconnect(client, userdata, rc):
        client.publish(f"/devices/{FLAGS.device_id}/state", 0, qos=1, retain=True)
        print("Disconnected")

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        client.publish(f"/devices/{FLAGS.device_id}/state", 1, qos=1, retain=True)
        print("Connected")

    @staticmethod
    def on_message(client, userdata, msg):
        print(msg.payload.decode())

    @staticmethod
    def on_publish(client, userdata, mid):
        """Paho callback when a message is sent to the broker."""
        print("on_publish")

    def init_cam(self):
        self.client_id = f"projects/{FLAGS.project_id}/locations/{FLAGS.cloud_region}/registries/{FLAGS.registry_id}/devices/{FLAGS.device_id}"
        print(f"Device client_id is '{self.client_id}'")
        client = mqtt.Client(client_id=self.client_id)
        client.username_pw_set(
            username="unused", password=utility.create_jwt(FLAGS.project_id, FLAGS.private_key_file, FLAGS.algorithm)
        )
        client.tls_set(ca_certs=FLAGS.google_mqtt_server_ca, tls_version=ssl.PROTOCOL_TLSv1_2)

        client.on_connect = Cam.on_connect
        client.on_disconnect = Cam.on_disconnect
        client.on_message = Cam.on_message
        client.on_publish = Cam.on_publish

        mqtt_config_topic = f"/devices/{FLAGS.device_id}/config"
        client.subscribe(mqtt_config_topic, qos=1)

        client.connect(FLAGS.mqtt_bridge_hostname, FLAGS.mqtt_bridge_port)
        return client


def select_cam_type():
    vid_cap_arg = None
    if FLAGS.cam_type == "webcam":
        print("Cam Type: Webcam")
        vid_cap_arg = 0
    elif FLAGS.cam_type == "ip_cam":
        vid_cap_arg = FLAGS.cam_link  # e.g. 'http://192.168.1.179:8080/video'
    elif FLAGS.cam_type == "video":
        vid_cap_arg = FLAGS.cam_link  # e.g. 'C:/tmp/test1.mkv'
    if vid_cap_arg is None:
        raise ValueError("Error: cam_type as arg is wrong")
    return vid_cap_arg


def resize_and_send(image, mqtt_topic, cam):
    image = cv2.resize(image, (FLAGS.input_size, FLAGS.input_size))
    _, img_encode = cv2.imencode('.jpg', image)
    data_encode = np.array(img_encode)
    str_encode = data_encode.tobytes()
    print(str_encode[0:50])
    cam.client.publish(mqtt_topic, str_encode)
    if FLAGS.show_stream:
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)
    time.sleep(FLAGS.interval_time)


def main(_argv):
    cam = Cam()
    mqtt_topic_img = f"/devices/{FLAGS.device_id}/events/image"

    vid = cv2.VideoCapture(select_cam_type())
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return_value, frame = vid.read()
    cam.img_height = frame.shape[0]
    cam.img_width = frame.shape[1]

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            raise ValueError("No stream")
        resize_and_send(frame, mqtt_topic_img, cam)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
else:
    pass
