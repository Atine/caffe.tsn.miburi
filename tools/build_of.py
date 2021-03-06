import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
import subprocess
import itertools
import argparse
out_path = ''


def dump_frames(vid_path):
    import cv2
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print 'dump_frames {} done'.format(vid_name)
    sys.stdout.flush()
    return file_list

def run_optical_flow_unpack(args):
    return run_optical_flow(*args)

def run_wrap_optical_flow_unpack(args):
    return run_wrap_optical_flow_unpack(*args)

def run_optical_flow(vid_path, vid_id, dev_id=0):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    
    if num_worker > 1:
        current = current_process()
        dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    subprocess.call([os.path.join(df_path)+'build/extract_gpu',
                     '-f', vid_path, 
                     '-x', flow_x_path,
                     '-y', flow_y_path,
                     '-i', image_path,
                     '-b', '20',
                     '-t', '1',
                     '-d', str(dev_id), 
                     '-o', out_format, 
                     '-w', str(new_size[0]), 
                     '-h', str( new_size[1])])

    #cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
    #    quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])
    #os.system(cmd)
    
    print 'Extracting flow: {} {} done'.format(str(vid_id).zfill(3), vid_name)
    sys.stdout.flush()
    return True


def run_warp_optical_flow(vid_path, vid_id, dev_id=0):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        vid_path, flow_x_path, flow_y_path, dev_id, out_format)

    os.system(cmd)
    print 'warp on {} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--df_path", type=str, default='./lib/dense_flow/', help='path to the dense_flow toolbox')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=2, help='number of GPU')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print "creating folder: "+out_path
        os.makedirs(out_path)

    vid_list = glob.glob(src_path+'/*.'+ext)
    print 'found {} videos'.format(len(vid_list))
    vid_id = xrange(len(vid_list))
    vid_tuple = list(zip(vid_list, vid_id))

    pool = Pool(num_worker)
    if flow_type == 'tvl1':
        pool.map(run_optical_flow_unpack, vid_tuple)
    elif flow_type == 'warp_tvl1':
        pool.map(run_warp_optical_flow_unpack, vid_tuple) 
