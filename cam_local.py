"""cam local

This module stores the main information about a camera stream.
First it will create a connection to a mqtt broker in the network and register there.
It takes the current frame and transform it suitable for the ml model.
The result can be viewed in window and will be sent to the given mqtt topic.

Includes count of frames with extra threshold (send only, if object was detected amount x in span y)
"""
import statistics
import time
import sys
import cv2
import paho.mqtt.client as mqtt
from absl import app, flags

import utility
import yolov4_tiny

FLAGS = flags.FLAGS
flags.DEFINE_string('cam_type', 'webcam', 'type of the video stream')
flags.DEFINE_string('cam_link', None,
                    'link to the video e.g. ip addr for ip_cam or path to video, for webcam not needed')
flags.DEFINE_string('broker_addr', 'localhost', 'address to the master broker in the network')
flags.DEFINE_integer('broker_port', 1883, 'port to the master broker in the network')
flags.DEFINE_boolean('show_stream', True, 'display the stream of camera in window')


class Cam:
    retry_time = 5  # Time in seconds when to resend cam activation request

    init = True  # Boolean to stop FPS calculation
    frame_times = []
    frame_time_max = 25  # Amount frames at beginning for fps calculation

    def __init__(self, mqtt_adr, name=None):
        self.mqtt_topics = utility.read_json(FLAGS.mqtt_config_file)
        self.broker_adr = mqtt_adr

        self.id = 0
        self.activated = False
        self.name = name
        self.ip = utility.get_primary_ip()
        self.uptime = utility.get_datetime()
        self.img_height = 0
        self.img_width = 0
        self.last_img = None

        self.activate_cam(broker_adr=mqtt_adr)
        self.topic = f"{self.mqtt_topics['device_root']}/{self.id}"

        self.client = mqtt.Client(str(self.id))
        self.client.on_disconnect = self.mqtt_on_disconnect
        self.client.will_set(f"{self.topic}/{self.mqtt_topics['device_status']}", f"{self.uptime}<:>{self.name}<:>0",
                             qos=1, retain=True)

        self.client.connect(mqtt_adr, FLAGS.broker_port, keepalive=20)
        self.client.publish(f"{self.topic}/{self.mqtt_topics['device_status']}", f"{self.uptime}<:>{self.name}<:>1",
                            retain=True)

    def mqtt_on_disconnect(self, client, userdata, msg):
        self.client.publish(f"{self.topic}/{self.mqtt_topics['device_status']}", f"{self.uptime}<:>{self.name}<:>0",
                            retain=True)

    def activate_cam(self, broker_adr="127.0.0.1"):
        try:
            cam_config = utility.read_json(FLAGS.cam_config_file)
            current_ip = utility.get_primary_ip()
            if cam_config['mqtt']['id'] != "" and cam_config['mqtt']['ip'] == current_ip:
                self.id = cam_config['mqtt']['id']
                self.name = cam_config['mqtt']['name']
                cam_config['mqtt']['ip'] = self.ip
                cam_config['mqtt']['uptime'] = self.uptime
                utility.write_json(cam_config, FLAGS.cam_config_file)
                self.activated = True
                return
            else:
                self.init_cam(broker_adr)
        except:
            print("Cam Config not found for current constellation")
            self.init_cam(broker_adr)

    # Create a temporary connection to mqtt to register the cam
    # After unique id from database is gathered disconnect tmp client
    def init_cam(self, broker_adr):
        client_tmp = mqtt.Client(clean_session=True)
        client_tmp.connect(broker_adr, FLAGS.broker_port)

        def on_message_client_tmp(client, userdata, msg):
            message = msg.payload.decode().split('<:>')
            msg_data = message[2].split(',')
            if str(client) == msg_data[0]:
                self.id = int(msg_data[1])
                self.activated = True

        client_tmp.subscribe(f"{self.mqtt_topics['device_cfg']['root']}/{self.mqtt_topics['device_cfg']['set_id']}")
        client_tmp.on_message = on_message_client_tmp
        client_tmp.loop_start()

        while not self.activated:
            client_tmp.publish(
                f"{self.mqtt_topics['device_cfg']['root']}/{self.mqtt_topics['device_cfg']['request_id']}",
                f"{utility.get_datetime()}<:>{client_tmp}")
            time.sleep(Cam.retry_time)

        client_tmp.loop_stop()

        if self.name is None:
            self.name = "cam" + str(self.id)
        client_tmp.publish(f"{self.mqtt_topics['device_cfg']['root']}/{self.mqtt_topics['device_cfg']['register']}",
                           f"{self.uptime}<:>{self.id}<:>{self.name}<:>1<:>{self.ip}", qos=1)
        print(f"Cam {self.id} activated")
        cam_json = {'mqtt': {
            'id': self.id,
            'name': self.name,
            'uptime': self.uptime,
            'ip': self.ip,
        }}
        utility.write_json(cam_json, FLAGS.cam_config_file)
        client_tmp.disconnect()
        return

    @staticmethod
    def calc_fps():
        fps = [(1 / x) for x in Cam.frame_times]
        fps_med = statistics.median(fps)
        Cam.init = False
        del fps
        del Cam.frame_times
        return int(fps_med)


def iteration(frame, interpreter, cam):
    prev_time = time.time()
    image_data = cv2.resize(frame, (FLAGS.input_size, FLAGS.input_size))
    image, _ = interpreter.iteration_step(frame, image_data, cam)
    print(cam.name)
    if FLAGS.show_stream:
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)
        #time.sleep(5)

    curr_time = time.time()
    exec_time = curr_time - prev_time
    if Cam.init:
        if len(Cam.frame_times) < Cam.frame_time_max:
            Cam.frame_times.append(exec_time)
        else:
            fps = Cam.calc_fps()
            recents = int(fps * yolov4_tiny.DetectedObject.min_time_detected)
            #else pop from empty list will happen
            if recents < 2:
                recents = 2
            yolov4_tiny.DetectedObject.recent = recents
            print(f"Objects are detected within {yolov4_tiny.DetectedObject.recent} frames")


def select_cam_type():
    vid_cap_arg = None
    if FLAGS.cam_type == "webcam":
        print("Cam Type: Webcam")
        vid_cap_arg = 0
    elif FLAGS.cam_type == "ip_cam":
        vid_cap_arg = FLAGS.cam_link  # e.g. 'http://192.168.1.179:8080/video'
    elif FLAGS.cam_type == "video":
        vid_cap_arg = FLAGS.cam_link  # e.g. 'C:/tmp/test.mkv'
    if vid_cap_arg is None:
        raise ValueError("Error: cam_type as arg is wrong")
    return vid_cap_arg


def main(_argv):
    cam = Cam(FLAGS.broker_addr)
    interpreter = yolov4_tiny.TfLiteInterpreter()
    print(interpreter.input_details)
    print(interpreter.output_details)

    vid = cv2.VideoCapture(select_cam_type())
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return_value, frame = vid.read()
    cam.img_height = frame.shape[0]
    cam.img_width = frame.shape[1]

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #For IR to Grayscale
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            raise ValueError("No stream")

        iteration(frame, interpreter, cam)


if __name__ == '__main__':
    print("Starting Cam")
    FLAGS(sys.argv)
    try:
        app.run(main)
    except SystemExit:
        pass
else:
    pass
