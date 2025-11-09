#-*- coding: utf-8 -*-
import argparse

model_config = argparse.ArgumentParser()

# model parameters
model_config.add_argument('--height', type=int, default=48, help='Input image height')
model_config.add_argument('--width', type=int, default=64, help='Input image width')
model_config.add_argument('--channel', type=int, default=3, help='Number of color channels')
model_config.add_argument('--ef_dim', type=int, default=12, help='Eye feature dimension')
model_config.add_argument('--agl_dim', type=int, default=2, help='Angle dimension')
model_config.add_argument('--encoded_agl_dim', type=int, default=16, help='Encoded angle dimension')

#demo
model_config.add_argument('--mod', type=str, default="flx", help='')
model_config.add_argument('--weight_set', type=str, default="weights", help='')
model_config.add_argument('--record_time', type=bool, default=False, help='')

model_config.add_argument('--tar_ip', type=str, default='localhost', help='Target IP address')
model_config.add_argument('--sender_port', type=int, default=5005, help='Sender socket port')
model_config.add_argument('--recver_port', type=int, default=5005, help='Receiver socket port')
model_config.add_argument('--uid', type=str, default='local', help='User ID')
model_config.add_argument('--P_IDP', type=float, default=6.3, help='Inter-pupillary distance parameter')
model_config.add_argument('--f', type=float, default=650, help='Focal length')
model_config.add_argument('--P_c_x', type=float, default=0, help='Camera position X')
model_config.add_argument('--P_c_y', type=float, default=-21, help='Camera position Y')
model_config.add_argument('--P_c_z', type=float, default=-1, help='Camera position Z')
model_config.add_argument('--S_W', type=float, default=62, help='Screen width')
model_config.add_argument('--S_H', type=float, default=35, help='Screen height')

def get_config():
    config, unparsed = model_config.parse_known_args()
    print(config)
    return config, unparsed
