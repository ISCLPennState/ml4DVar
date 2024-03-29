import cv2
import os
import sys
import glob

save_name = sys.argv[1]

root_dir = '/eagle/MDClimSim/mjp5595/data/stormer/'
analysis_dir = os.path.join(root_dir,save_name,'plots','analysis')
bg_vs_ana_dir = os.path.join(root_dir,save_name,'plots','bg_vs_ana')

#ana_files = os.listdir(analysis_dir)
ana_files = glob.glob(os.path.join(analysis_dir,'*.png'))
ana_files.sort()
#print('ana_files :',ana_files)
#ana_files = ana_files[:50]

ana_save_name = os.path.join(root_dir,save_name,'plots','analysis',save_name+'.mp4')
print()
print('ana_save_name :',ana_save_name)

frames_per_second = 6.0
max_size = 2047
big_side = 2047
r, c = None, None
for j,gif_f in enumerate(ana_files):
    frame = cv2.imread(gif_f)
    
    if r is None:
        r,c,_ = frame.shape
        if max(r,c) > max_size:
            big_side = max(r,c)
        c = int(c*(max_size/big_side))
        r = int(r*(max_size/big_side))

        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(ana_save_name,
                                    fourcc,
                                    frames_per_second,
                                    (c,r),
                                    )

    writer.write(cv2.resize(frame,(c,r)))
    if j == 0 or j == (len(ana_files)-1):
        for _ in range(int(frames_per_second)):
            writer.write(cv2.resize(frame,(c,r)))
writer.release()