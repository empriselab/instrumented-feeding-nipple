import rosbag  # use pip install bagpy
import numpy as np
from PIL import Image
import os
import time
import argparse

from rich.progress import track
from rich.style import Style


forque_sensor_names = ['/ft_sensor/netft_data']


FORQUE_ENABLE = True  # True
JOINT_ENABLE = True  # True
COLOR_ENABLE = True
DEPTH_ENABLE = False  # True

SKIP_EXISTING = True


# parse the --dir or -d as the target directory list, default to current directory
# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dir', type=str, nargs='+', default=[os.getcwd()], help='the directories of bag files')
# args = parser.parse_args()
# home = os.getcwd()


def get_targets_names(path):
    # get all current dirtory files, filter out not .bag file, return the list
    # return ['transfer_example.bag']
    targets = []
    for file in os.listdir(path):
        if file.endswith('.bag') and os.path.isfile(os.path.join(path, file)):
            targets.append(file)
    return targets


def save_forque(msg, f, start_time):
    '''
    csv format:
    time,fx,fy,fz,tx,ty,tz
    '''
    current_time = msg.timestamp.to_sec() - start_time
    fx = msg.message.wrench.force.x
    fy = msg.message.wrench.force.y
    fz = msg.message.wrench.force.z
    tx = msg.message.wrench.torque.x
    ty = msg.message.wrench.torque.y
    tz = msg.message.wrench.torque.z
    f.write('{:.04f},{},{},{},{},{},{}\n'.format(current_time, fx, fy, fz, tx, ty, tz))


if __name__ == '__main__':

    # parse the --dir or -d as the target directory, default to current directory
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=os.getcwd(), help='the directory of bag files')
    args = parser.parse_args()
    working_dir = args.dir

    targets = get_targets_names(working_dir)
    total_num = len(targets)
    count = 0
    print(str(targets) + '\n\n')
    for target in targets:
        count += 1
        print('working on bag file {}/{}: {}'.format(count, total_num, target))
        bag = rosbag.Bag(os.path.join(working_dir, target))
        # make a directory for the images, removing the .bag extension
        start_time = bag.get_start_time()
        output_dir_name = target[:-4]
        output_dir_full_path = os.path.join(working_dir, output_dir_name)
        # test if dir already exists, if so, skip
        if SKIP_EXISTING and os.path.exists(output_dir_full_path):
            print('Skipping existing directory: {}'.format(output_dir_name))
            continue
        os.makedirs(output_dir_full_path)

        if FORQUE_ENABLE:
            for forque_name in forque_sensor_names:
                if forque_name not in bag.get_type_and_topic_info()[1].keys():
                    print('Skipping non-existing topic: {} in bag file {}'.format(forque_name, target))
                    continue
                message_count = bag.get_message_count(forque_name)
                if message_count == 0:
                    print('Skipping empty topic: {} in bag file {}'.format(forque_name, target))
                    continue
                file_name = forque_name.replace('/', '_') + '.csv'
                file_name = file_name[1:] if file_name[0] == '_' else file_name
                file_path = os.path.join(output_dir_full_path, file_name)
                topic = bag.read_messages(topics=forque_name)
                with open(file_path, 'w') as f:
                    f.write('time,fx,fy,fz,tx,ty,tz\n')
                    for msg in track(topic, total=message_count, description='Bag {}/{}, processing {} in {}'.format(count, total_num, forque_name, target), finished_style=Style(color='green')):
                        save_forque(msg, f, start_time)

        bag.close()
