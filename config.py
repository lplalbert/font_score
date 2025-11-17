#整体

image_path = '/home/ylyu/workspace/text_score/image/test1_768.jpg'  # 田字格图片
target_ttf = '/mnt/ylyu/text_data/ttf_tar/华栋正楷第三版 Regular.ttf'  # 替换为target.png文件的路径
rows = 4  # 例如10行
cols = 8   # 例如8列

save_temp=True
#gravity
gravity_weight1=0.7
gravity_weight2=0.3

#shape
area_weight=0.7
wh_weight=0.3

#skeleton
ske_vgg_weight=0.3
ske_ssim_weight=0.4
ske_hasdf_weight=0.3

#hull
hull_sim_weight=0.6
hull_dis_weight=0.4

#total
base_weight=0.15
grid_weight=0.35
ske_weight=0.3
hull_weight=0.2