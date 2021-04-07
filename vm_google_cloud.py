"""vm cloud

This module is run by a virtual machine in google cloud engine.
It will connect to the Pub/Sub of the camera.
Also a connection to the MQTT Bridge is established. (not necessary needed)
The module handles the receiving of the image, prediction and do entries in the google sql database.
"""
import os
import sys
import time
import numpy as np
import cv2
import paho.mqtt.client as mqtt
from absl import app, flags
from paho.mqtt.client import ssl
import threading
import mariadb
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

import utility
import yolov4_tiny

FLAGS = flags.FLAGS
flags.DEFINE_string('project_id', 'smart-cam-ba', 'project id of google cloud platform')
flags.DEFINE_string('cloud_region', 'europe-west1', 'cloud region of the google IoT Core registry')
flags.DEFINE_string('registry_id', 'cam', 'registry id in the google IoT Core')
flags.DEFINE_string('abo_id', 'cam_abo', 'abo id for the pub/sub theme')
flags.DEFINE_string('device_id', 'vm-instance', 'device id that is already added in the corresponding registry id')
flags.DEFINE_string('algorithm', 'RS256', 'encryption algorithm to use for jwt')
flags.DEFINE_string('private_key_file', './data/rsa_private.pem', 'file path to the private key .pem')
flags.DEFINE_string('google_mqtt_server_ca', './data/roots.pem',
                    'file path to the google root CA certification package')
flags.DEFINE_string('mqtt_bridge_hostname', 'mqtt.googleapis.com', 'MQTT bridge hostname')
flags.DEFINE_integer('mqtt_bridge_port', 8883, 'MQTT bridge port')

flags.DEFINE_string('mariadblogin', './data/mariadb_config.json', 'file path to the mariadb login data')
flags.DEFINE_string('det_table', 'detections', 'name of the table containing all detections')


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './data/smart-cam-ba-78c8e2da3568.json'


class Prediction:
    def __init__(self, cam, db):
        self.queue_img = []
        self.queue_dev_info = []
        self.interpreter = yolov4_tiny.TfLiteInterpreter()
        print(self.interpreter.input_details)
        print(self.interpreter.output_details)

        s = threading.Thread(target=self.make_predictions, args=(cam,db))
        s.start()

    def make_predictions(self, cam, db):
        while True:
            if self.queue_img:
                image = self.queue_img.pop(0)
                image, objs_found = self.interpreter.iteration_step(image, image, cam, publish=False)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                data = self.queue_dev_info.pop(0)
                if objs_found is not None:
                    for obj in objs_found:
                        path = f"./images/{obj[4]}"
                        img_path_abs = os.path.abspath(path)
                        cv2.imwrite(path, result)
                        item = [obj[0], obj[1], obj[2], obj[3], img_path_abs, int(data[1])]
                        print(item)
                        db.insert_item(item, FLAGS.det_table)
                        utility.send_notification(obj[4])


def receive_messages(cam, db, timeout=None):
    """Receives messages from a pull subscription."""
    pred = Prediction(cam, db)
    subscriber = pubsub_v1.SubscriberClient()

    subscription_path = subscriber.subscription_path(FLAGS.project_id, FLAGS.abo_id)

    def callback(message):
        print(f"Received {message}.")
        message.ack()
        if message.attributes["subFolder"] == "image":
            nparr = np.frombuffer(message.data, np.uint8)
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            pred.queue_img.append(img_decode)
            device_info = [message.attributes["deviceId"],
                           message.attributes["deviceNumId"],
                           message.attributes["deviceRegistryId"],
                           message.attributes["deviceRegistryLocation"],
                           message.attributes["projectId"]]
            pred.queue_dev_info.append(device_info)

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}..\n")

    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result()
            time.sleep(5)
        except TimeoutError:
            streaming_pull_future.cancel()


class Cam:
    def __init__(self, name):
        self.id = 0
        self.activated = False
        self.name = name
        self.ip = utility.get_primary_ip()
        self.uptime = utility.get_datetime()
        self.img_height = FLAGS.input_size
        self.img_width = FLAGS.input_size
        self.last_img = None


class Mqtt:
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
    def on_message(unused_client, unused_userdata, message):
        """Callback when the device receives a message on a subscription."""
        payload = str(message.payload.decode("utf-8"))
        print(
            "Received message '{}' on topic '{}' with Qos {}".format(
                payload, message.topic, str(message.qos)
            )
        )

    @staticmethod
    def on_publish(client, userdata, mid):
        """Paho callback when a message is sent to the broker."""
        print("on_publish")

    def init_cam(self):
        self.client_id = f"projects/{FLAGS.project_id}/locations/{FLAGS.cloud_region}/registries/{FLAGS.registry_id}/devices/{FLAGS.device_id}"
        print(f"Device client_id is '{self.client_id}'")
        client = mqtt.Client(client_id=self.client_id)
        client.username_pw_set(
            username="unused",
            password=utility.create_jwt(FLAGS.project_id, FLAGS.private_key_file, FLAGS.algorithm)
        )
        client.tls_set(ca_certs=FLAGS.google_mqtt_server_ca, tls_version=ssl.PROTOCOL_TLSv1_2)

        client.on_connect = Mqtt.on_connect
        client.on_disconnect = Mqtt.on_disconnect
        client.on_message = Mqtt.on_message
        client.on_publish = Mqtt.on_publish

        client.connect(FLAGS.mqtt_bridge_hostname, FLAGS.mqtt_bridge_port)

        mqtt_config_topic = f"/devices/{FLAGS.device_id}/config"
        client.subscribe(mqtt_config_topic, qos=1)

        mqtt_command_topic = f"/devices/{FLAGS.device_id}/commands/#"
        client.subscribe(mqtt_command_topic, qos=0)

        return client


class Database:
    @staticmethod
    def connect_mariadb():
        data = utility.read_json(FLAGS.mariadblogin)
        try:
            conn = mariadb.connect(
                user=data["user"],
                password=data["password"],
                host=data["host"],
                database=data["database"]
            )
        except mariadb.Error as e:
            print(f"\nError connecting to MariaDB Platform: {e}")
            sys.exit(1)
        return conn

    def get_column(self, column, table, orderc=None):
        conn = self.connect_mariadb()
        cur = conn.cursor()
        items = []
        try:
            sql = f"SELECT {column} FROM {table}"
            if orderc is not None:
                sql = sql + f" ORDER BY {orderc}"
            cur.execute(sql)
            for item in cur:
                items.append(item[0])
        except mariadb.Error as e:
            print(f"\nError connecting to MariaDB Platform: {e}")
            sys.exit(1)
        conn.close()
        return items

    def insert_item(self, item, table):
        conn = self.connect_mariadb()
        cur = conn.cursor()
        try:
            cur.execute(
                f"INSERT INTO {table} (name, id_object, probability, timestamp, image_path, id_cam) VALUES "
                f"(%s, %s, %s, %s, %s, %s)", (item[0], item[1], item[2], item[3], item[4], item[5]))
            conn.commit()
        except NameError:
            print(f"Table Name is not defined")
        except mariadb.Error as e:
            print(f"\nError connecting to MariaDB Platform: {e}")
            sys.exit(1)

        conn.close()
        return



def main(_argv):
    print("Starting VM")

    mqtt = Mqtt()
    cam = Cam(mqtt.client_id)
    db = Database()
    receive_messages(cam, db)
    while True:
        time.sleep(1)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
else:
    pass
