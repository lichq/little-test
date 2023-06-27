# %%
import numpy as np
import math
from mpl_toolkits import mplot3d
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as lines
myfont = fm.FontProperties(fname='simsun.ttc')
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
# from icp import icp

def Rot(theta):
    theta = (theta/180*np.math.pi)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def load_txt(input_path, proj_path='./', type = 0):
    whole_path = os.path.join('./', input_path)
    res = np.loadtxt(whole_path)
    if type == 1:
        xyz = res[:, 0:3]
    else:
        xyz = res[:, 1:4]
   
    # preprocess of our method
    if type == 1:    
        our_pose = []
        opose = -xyz[0]
        for i in range(0, len(xyz)):
            opose -= xyz[i]
            our_pose.append(np.copy(opose))
        our_pose = np.array(our_pose)
        xyz = our_pose

    # preprocess of original method
    elif type == 2:
        pose = []
        global_t = np.identity(4)
        last_line = np.array([0, 0, 0, 1])
        for i in range(0, len(xyz)):
            t = np.array(res[i, 1:4]) # x y z
            Rq = res[i, 4:] # quat
            tmp = Rq[0]
            Rq[0] = Rq[3]
            Rq[3] = tmp
            r = np.array(R.from_quat(Rq).as_matrix())
            T_temp = np.c_[r, t]
            T = np.row_stack((T_temp, last_line))
            global_t = np.dot(T, global_t)
            pose.append(np.copy(global_t[:3, 3]).T)
        xyz = np.array(pose)  
    
    length = 0
    for i in range(1, len(xyz)):
        length += math.sqrt((xyz[i-1][0] - xyz[i][0])**2 + (xyz[i][1] - xyz[i-1][1])**2)

    return xyz, length

# draw the trajectory accoding to the tf files

def scaleAndfilt(gt_length, arr_length, input_arr, circle = False):
    arr = input_arr / (arr_length/gt_length)
    arr[:,1] *= -1
    # if circle == True:
    #     arr[:,0] *= -1

    return  arr

def show_trajectory(proj_path='./', D3=False):

    gt_xyz, len_gt = load_txt('gt/gt_real.txt')
    our_xyz, len_our = load_txt('J/Areal/tf_opt.txt', type = 1)
    our_xyz1, len_our1 = load_txt('J/Areal/tf_base.txt', type = 1)
    ori_xyz, len_ori = load_txt('ori/resultsMul_real.txt', type = 2)
    fmt_xyz, len_fmt = load_txt('ori/result_real.txt', type = 2)
    orb_xyz, len_orb = load_txt('orb/orb_real.txt')
    dso_xyz, len_dso = load_txt('dso/result_strange.txt')

    len_gt /= len(gt_xyz)
    len_our /= len(our_xyz)
    len_our1 /= len(our_xyz1)
    len_ori /= len(ori_xyz)
    len_fmt /= len(fmt_xyz)
    len_orb /= len(orb_xyz)
    len_dso /= len(dso_xyz)



    our_xyz = scaleAndfilt(len_gt, len_our, our_xyz, True)
    our_xyz1 = scaleAndfilt(len_gt, len_our1, our_xyz1, True)
    ori_xyz = scaleAndfilt(len_gt, len_ori, ori_xyz, True)
    fmt_xyz = scaleAndfilt(len_gt, len_fmt, fmt_xyz, True)
    orb_xyz = scaleAndfilt(len_gt, len_orb, orb_xyz)
    dso_xyz = scaleAndfilt(len_gt, len_dso, dso_xyz, False)

    our_xyz += gt_xyz[0] - our_xyz[0]
    our_xyz1 += gt_xyz[0] - our_xyz1[0] 
    # our_xyz[220:, 0] += -1
    # our_xyz1[0] = gt_xyz[0]
    ori_xyz += gt_xyz[0] - ori_xyz[0]
    fmt_xyz += gt_xyz[0] - fmt_xyz[0]
    dso_xyz += gt_xyz[118] - dso_xyz[0]

    
    gt = gt_xyz
    gt = np.transpose(Rot(17.3) @ np.transpose(gt[:,:2]))

    # _, our_xyz = icp(gt[:, :2], our_xyz[:, :2], verbose=True)
    # _, ori_xyz = icp(gt[:, :2], ori_xyz[:, :2], verbose=True)
    # _, fmt_xyz = icp(gt[:, :2], fmt_xyz[:, :2], verbose=True)
    # _, orb_xyz = icp(gt[:, :2], orb_xyz[:, :2], verbose=True)
    # _, dso_xyz = icp(gt[:, :2], dso_xyz[:, :2], verbose=True)

    fig = plt.figure(1, figsize=(6, 5.2), dpi=300)
    ax3 = fig.add_subplot()

    ax3.scatter(gt[:,0], gt[:,1], c='red', s=1)
    ax3.scatter(gt[::50,0], gt[::50,1], c='red', s=4)
    ax3.scatter(our_xyz[:, 0], our_xyz[:, 1], c='c', s=1)
    ax3.scatter(our_xyz[::50, 0], our_xyz[::50, 1], c='c', s=4)
    # ax3.scatter(our_xyz1[:, 0], our_xyz1[:, 1], c='orange', s=1)
    # ax3.scatter(our_xyz1[::50, 0], our_xyz1[::50, 1], c='orange', s=4)
    ax3.scatter(ori_xyz[:, 0], ori_xyz[:, 1], c='green', s=1)
    ax3.scatter(ori_xyz[::50, 0], ori_xyz[::50, 1], c='green', s=4)
    ax3.scatter(fmt_xyz[:, 0], fmt_xyz[:, 1], c='blue', s=1)
    ax3.scatter(fmt_xyz[::50, 0], fmt_xyz[::50, 1], c='blue', s=4)
    ax3.scatter(orb_xyz[:, 0], orb_xyz[:, 1], c='greenyellow', s=1)
    ax3.scatter(orb_xyz[::50, 0], orb_xyz[::50, 1], c='greenyellow', s=4)
    ax3.scatter(dso_xyz[:, 0], dso_xyz[:, 1], c='m', s=1)
    ax3.scatter(dso_xyz[::50, 0], dso_xyz[::50, 1], c='m', s=4)


    ax3.plot(gt[:,0], gt[:,1], 'red', label="GT", linewidth=0.8)
    ax3.plot(our_xyz[:, 0], our_xyz[:, 1], 'c', label="o-eFMT",  linewidth=0.8)
    # ax3.plot(our_xyz1[:, 0], our_xyz1[:, 1], 'orange', label="our-eFMT",  linewidth=0.8)
    ax3.plot(ori_xyz[:, 0], ori_xyz[:, 1], 'green', label="eFMT",  linewidth=0.8)
    ax3.plot(fmt_xyz[:, 0], fmt_xyz[:, 1], 'blue', label="FMT",  linewidth=0.8)
    ax3.plot(orb_xyz[:, 0], orb_xyz[:, 1], 'greenyellow', label="ORB-SLAM",  linewidth=0.8)
    # ax3.plot(dso_xyz[:, 0], dso_xyz[:, 1], 'm', label="DSO",  linewidth=0.8)
    # ax3.set_title("Analemma trajectories align in 2D")

    ax3.set_xlim(xmin = -110, xmax = 100)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.legend(loc='lower left', fontsize = 'small')
    # fig.show()
    fig.savefig(os.path.join(proj_path, 'output', "trajectory2D_8.pdf"))
    print("-------------")
    # print(our_xyz[:, :2])
    # print("")
    show_error_boxFigure(our_xyz[:, :2], our_xyz1[:, :2], ori_xyz[:, :2], fmt_xyz[:, :2], orb_xyz[:, :2], dso_xyz[:, :2], gt)

def get_error(gt, arr, type = 0):



    arr_error = []
    if type == 1:
        for i in range(0, int(len(arr))):
            absolute_err = math.sqrt((gt[i+70][0] - arr[i][0])**2 + (gt[i+70][1] - arr[i][1])**2)
            arr_error.append(absolute_err)
    else:
        for i in range(0, int(len(arr))):
            absolute_err = math.sqrt((gt[i][0] - arr[i][0])**2 + (gt[i][1] - arr[i][1])**2)
            # print(absolute_err, gt[i], arr[i], i)
            arr_error.append(absolute_err)
    np_err = np.array(arr_error)
    return np_err


def show_error_boxFigure(our, our1, ori, fmt, orb, dso, gt):

    our_err = get_error(gt, our)
    our_err1 = get_error(gt, our1)
    ori_err = get_error(gt, ori)
    fmt_err = get_error(gt, fmt)
    orb_err = get_error(gt, orb)
    dso_err = get_error(gt, dso, type=1)
    print("Method: our_opt, our_base, ori, fmt, orb")
    print("max: ", np.round(np.max(our_err), 2), np.round(np.max(our_err1), 2), np.round(np.max(ori_err), 2), np.round(np.max(fmt_err), 2), np.round(np.max(orb_err), 2))
    print("mean: ", np.round(np.mean(our_err), 2), np.round(np.mean(our_err1), 2), np.round(np.mean(ori_err), 2), np.round(np.mean(fmt_err), 2), np.round(np.mean(orb_err), 2))
    print("median: ", np.round(np.median(our_err), 2), np.round(np.median(our_err1), 2), np.round(np.median(ori_err), 2), np.round(np.median(fmt_err), 2), np.round(np.median(orb_err), 2))

    data = {
        'o-eFMT': our_err,
        # 'our-eFMT': our_err1,
        'eFMT': ori_err,
        'FMT': fmt_err,
        'ORB-SLAM3': orb_err,
        # 'DSO': dso_err,
    }
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()

    plt.figure(figsize=(5, 5), dpi = 300)

    df.plot.box(showfliers = False,title="Real trajectory translation error", figsize=(6, 5.2))
    

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.ylabel("translation error [m]")
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()
    plt.savefig(os.path.join("./", 'output', "boxerror_c.pdf"))
  

if __name__ == "__main__":
    show_trajectory()
    # show_error_boxFigure()
    pass


# %%
