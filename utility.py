"""utility

This module just includes some usefull functions.
"""
import datetime
import json
import socket
import time
import re
import jwt
from absl import flags
import requests
import urllib.request
from urllib.error import HTTPError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

FLAGS = flags.FLAGS
flags.DEFINE_string('cam_config_file', './data/cam_config.json', 'path to the cam_config')
flags.DEFINE_string('mqtt_config_file', './data/mqtt_config.json', 'path to the mqtt_config')
flags.DEFINE_string('django_config_file', './data/django_config.json', 'path to the django_config')


def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)
    return 0


def get_datetime(file_format=False):
    if not file_format:
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')


def read_info(file_name):
    infos = {}
    with open(file_name, 'r') as data:
        for ID, info in enumerate(data):
            infos[ID] = info.strip('\n')
    return infos


def get_primary_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def create_jwt(project_id, private_key_file, algorithm):
    """Creates a JWT (https://jwt.io) to establish an MQTT connection.
        Args:
         project_id: The cloud project ID this device belongs to
         private_key_file: A path to a file containing either an RSA256 or
                 ES256 private key.
         algorithm: The encryption algorithm to use. Either 'RS256' or 'ES256'
        Returns:
            A JWT generated from the given project_id and private key, which
            expires in 20 minutes. After 20 minutes, your client will be
            disconnected, and a new JWT will have to be generated.
        Raises:
            ValueError: If the private_key_file does not contain a known key.
    """
    token = {
        # The time that the token was issued at
        "iat": datetime.datetime.utcnow(),
        # The time the token expires.
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=1440),
        # The audience field should always be set to the GCP project id.
        "aud": project_id,
    }
    # Read the private key file.
    with open(private_key_file, "r") as f:
        private_key = f.read()
    print(
        "Creating JWT using {} from private key file {}".format(
            algorithm, private_key_file
        )
    )
    return jwt.encode(token, private_key, algorithm=algorithm)


def login_to_django(ip):
    try:
        cfg = read_json(FLAGS.django_config_file)
        client = requests.session()
        url = ip + cfg['login_url']
        client.get(url)
        if 'csrftoken' in client.cookies:
            csrftoken = client.cookies['csrftoken']
        else:
            csrftoken = client.cookies['csrf']
        login_data = dict(username=cfg['username'], password=cfg['password'], csrfmiddlewaretoken=csrftoken,
                          next=cfg['next_url'])
        r = client.post(url, data=login_data, headers=dict(Referer=url))
        print(r)
    except:
        pass


def send_notification(img_name):
    try:
        cfg = read_json(FLAGS.django_config_file)
        if cfg['ip'] == '':
            ip = "http://" + get_primary_ip()
            ip += ':' + cfg['port']
        else:
            ip = cfg['ip']
        login_to_django(ip)
        print(ip, cfg['port'])
        link = ip + cfg['update_img']
        urllib.request.urlopen(link)
        img_link = ip + f"/media/" + img_name
        email = cfg['email']
        password = cfg['email_password']
        send_to_email = cfg['send_to']
        subject = "Object Detected"
        message = f"An Object has been detected\n" \
                  f"View at {img_link}"
        msg = MIMEMultipart()
        msg["From"] = email
        msg["To"] = send_to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, 'plain'))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email, password)
        text = msg.as_string()
        server.sendmail(email, send_to_email, text)
        server.quit()
    except:
        pass


class REMatcher(object):
    def __init__(self, matchstring):
        self.matchstring = matchstring

    def match(self, regexp):
        self.rematch = re.match(regexp, self.matchstring)
        return bool(self.rematch)

    def group(self, i):
        return self.rematch.group(i)
