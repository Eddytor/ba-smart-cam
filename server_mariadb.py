"""server mariadb

This module handles all the local storing in the database. It will connect as a MQTT client.
A connection to django is also created for transmitting the image.
It will also send the notification of a detection.
"""
import sys
import time
import paho.mqtt.client as mqtt
from absl import app
from absl import flags
import mariadb
import cv2
import numpy as np
import os


import utility

FLAGS = flags.FLAGS
flags.DEFINE_string('cam_table', 'cameras', 'name of the table containing cameras information')
flags.DEFINE_string('det_table', 'detections', 'name of the table containing all detections')

flags.DEFINE_string('image_path', './images', 'path where to store detection images')
flags.DEFINE_string('mariadb_config', './data/mariadb_config.json', 'file path to the mariadb login data')


class Mqtt:
    def __init__(self, mqtt_adr):
        self.mqtt_topics = utility.read_json(FLAGS.mqtt_config_file)

        self.broker_adr = mqtt_adr
        self.name = "server"
        self.db = Database()
        self.client = mqtt.Client(self.name)
        self.client.connect(mqtt_adr)
        self.assign_cams()
        self.client.loop_start()
        while True:
            time.sleep(7)

    def assign_cams(self):

        def on_message(client, userdata, msg):

            m = utility.REMatcher(msg.topic)

            # Register a camera by finding a free id
            if msg.topic == f"{self.mqtt_topics['device_cfg']['root']}/{self.mqtt_topics['device_cfg']['request_id']}":
                message = msg.payload.decode().split('<:>')
                ids_occupied = self.db.get_column("id", FLAGS.cam_table, orderc="id")
                new_id = 0
                for cam_id in ids_occupied:
                    if new_id != cam_id:
                        break
                    else:
                        new_id = new_id + 1
                self.db.insert_item([new_id, None, 0], FLAGS.cam_table)
                client.publish(f"{self.mqtt_topics['device_cfg']['root']}/{self.mqtt_topics['device_cfg']['set_id']}",
                               f"{utility.get_datetime()}<:>{client}<:>{message[1]},{new_id}")

            # Receive the additional information after registration
            elif msg.topic == f"{self.mqtt_topics['device_cfg']['root']}/{self.mqtt_topics['device_cfg']['register']}":
                message = msg.payload.decode().split('<:>')
                self.db.update_all_items(message, message[1], "id", FLAGS.cam_table)

            # Change the device status of a camera
            elif m.match(rf"^{self.mqtt_topics['device_root']}/([^\s]+)/{self.mqtt_topics['device_status']}"):
                message = msg.payload.decode().split('<:>')
                self.db.update_item("status", (message[2], m.group(1)), "id", FLAGS.cam_table)
                self.db.update_item("uptime", (message[0], m.group(1)), "id", FLAGS.cam_table)

            # Receive the information of a detection
            elif m.match(rf"^{self.mqtt_topics['device_root']}/([^\s]+)/{self.mqtt_topics['detection_info']}/([^\s]+)"):
                message = msg.payload.decode().split('<:>')
                img_path_rel = FLAGS.image_path + "/" + message[2]
                img_path_abs = os.path.abspath(img_path_rel)

                ids_occupied = self.db.get_column("id", FLAGS.det_table, orderc="id")
                new_id = 0
                for cam_id in ids_occupied:
                    if new_id != cam_id:
                        break
                    else:
                        new_id = new_id + 1
                item = [new_id, m.group(2), message[0], message[1], message[3], img_path_abs, m.group(1)]
                self.db.insert_item(item, FLAGS.det_table)
                print(msg.topic, message)

            # Receive the image of a detection and send notification
            elif m.match(rf"^{self.mqtt_topics['device_root']}/([^\s]+)/{self.mqtt_topics['image']}/([^\s]+)"):
                nparr = np.frombuffer(msg.payload, np.uint8)
                img_decode = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                path = f"{FLAGS.image_path}/{m.group(2)}"
                cv2.imwrite(path, img_decode)
                utility.send_notification(m.group(2))

        self.client.subscribe(f"{self.mqtt_topics['device_cfg']['root']}/#")
        self.client.subscribe(f"{self.mqtt_topics['device_root']}/#")
        self.client.on_message = on_message


class Database:
    @staticmethod
    def connect_mariadb():
        data = utility.read_json(FLAGS.mariadb_config)
        try:
            conn = mariadb.connect(
                user=data["user"],
                password=data["password"],
                host=data["host"],
                port=data["port"],
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
            print(f"\nMariaDB Error: {e}")
        conn.close()
        return items

    def insert_item(self, item, table):
        conn = self.connect_mariadb()
        cur = conn.cursor()
        try:
            if table == FLAGS.cam_table:
                cur.execute(
                    f"INSERT INTO {table} (id, name, status) VALUES (%s, %s, %s)", (item[0], item[1], item[2]))
            elif table == FLAGS.det_table:
                cur.execute(
                    f"INSERT INTO {table} (id, name, id_object, probability, timestamp, image_path, id_cam) VALUES "
                    f"(%s, %s, %s, %s, %s, %s, %s)", (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))
            conn.commit()
        except mariadb.Error as e:
            print(f"\nMariaDB Error: {e}")
        except NameError:
            print(f"Table Name is not defined")
        except:
            e = sys.exc_info()[0]
            print(e)

        conn.close()
        return

    def update_all_items(self, content, match, column, table):
        conn = self.connect_mariadb()
        cur = conn.cursor()
        try:
            if table == FLAGS.cam_table:
                val = (content[0], content[2], content[3], content[4], match)
                sql = f"UPDATE {table} SET uptime = %s, name = %s, status = %s, ip = %s WHERE {column} = %s"
                cur.execute(sql, val)
            conn.commit()
        except mariadb.Error as e:
            print(f"\nMariaDB Error: {e}")

        conn.close()
        return

    def update_item(self, val, content, column, table):
        conn = self.connect_mariadb()
        cur = conn.cursor()
        try:
            if table == FLAGS.cam_table:
                sql = f"UPDATE {table} SET {val} = %s WHERE {column} = %s"
                cur.execute(sql, content)
            conn.commit()
        except mariadb.Error as e:
            print(f"\nMariaDB Error: {e}")

        conn.close()
        return


def main(_argv):
    print("Starting Server")
    Mqtt("127.0.0.1")
    return


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
else:
    pass
