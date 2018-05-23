
#!/usr/bin/env python

"""Code for generation noise for MIBURI dataset
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import time
from functools import wraps

import random


NUMBER_FRAMES = 120


def parse_miburi_splits():
    class_ind = [x.strip().split() for x in open('/home/mil/chou/pytorch.tsn.miburi/miburi_splits/classInd.txt')]
    label_mapping = {x[1]:int(x[0]) for x in class_ind}

    def line2rec(line):
        item = line.strip().split('-')
        label = label_mapping[items[0]]
        vid = line.strip().split(' ')[0].split('.')[0]
        return vid, label

    splits = []
    train_list = [line2rec(x) for x  in open('/home/mil/chou/pytorch.tsn.miburi/miburi_splits/trainlist.txt')]
    test_list = [line2rec(x) for x  in open('/home/mil/chou/pytorch.tsn.miburi/miburi_splits/testlist.txt')]
    splits.append((train_list, test_list))
    return splits


def Timeit(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        result = f(*args, **kwds)
        print("in {1:.6f} seconds, {0}() excuted. ".format(f.__name__, (time.time()-start_time)))
        return result
    return wrapper


"""Helpers"""
@Timeit
def make_border_list():
    border_points = []
    # hand-crafted approach
    for y in range(480):
        border_points.append((0,y))
        border_points.append((640,y))
    for x in range(640):
        border_points.append((x,0))
        border_points.append((x,480))
    return border_points
borderlist = make_border_list()

@Timeit
def random_border():
    
    init_point = random.choice(borderlist)
    end_point = random.choice(borderlist)
    while (end_point[0] == init_point[0] or end_point[1] == init_point[1]):
        end_point = random.choice(borderlist)
    
    print init_point
    print end_point
    return (init_point, end_point)


@Timeit
def line_with_interval():
    
    init, end = random_border()
    min_x = min(init[0],end[0])
    max_x = max(init[0],end[0])
    print init
    print end
    
    
    # Define the known points
    temp_x = [init[0], end[0]]
    temp_y = [init[1], end[1]]

    coefficients = np.polyfit(temp_x, temp_y, 1)
    polynomial = np.poly1d(coefficients)
    output_x = np.linspace(min_x,max_x,num=NUMBER_FRAMES)
    output_y = polynomial(output_x)
    
    '''plotting
    plt.plot(output_x, output_y)
    plt.plot( temp_x[0], temp_y[0], 'go' )
    plt.plot( temp_x[1], temp_y[1], 'go' )
    plt.grid('on')
    
    axes = plt.gca()
    axes.set_xlim([0,640])
    axes.set_ylim([0,480])
    plt.show()
    '''
    return output_x, output_y
        
    


fig = plt.figure()
fig.set_dpi(100)
ax = plt.axes(xlim=(0, 640), ylim=(0, 480))
center = (100,100)
patch = plt.Circle(center, 20, fc='y')

class GenerateNoise(object):
    def __init__(self):
        pass


def init():
    patch.center = center
    ax.add_patch(patch)
    return patch,

def animate_circle(i):
    x, y = patch.center
    x = center[0] + 100 * np.sin(np.radians(i))
    y = center[1] + 100 * np.cos(np.radians(i))
    patch.center = (x, y)
    return patch,


x_list, y_list = line_with_interval()
def animate_line(i):
    x, y = x_list[i], y_list[i]
    patch.center = (x, y)
    #print x,y
    return patch,


def main():
    #plt.show()
    anim = animation.FuncAnimation(fig, animate_line, 
                                   init_func=init, 
                                   frames=120, 
                                   interval=20,
                                   blit=True)

    anim.save('animation.mp4', fps=29, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])

if __name__ == '__main__':
    main()

