# 机器人避障寻径模型训练和自动标注
## mlp_model_train可以对MLP模型进行训练
python mlp_model_train.py --path dir_path --name mp4_name   
dir_path：数据所在的路径名   
mp4_name: mp4文件名，不包括后缀.mp4   
   
需要vgg16_label_．．．的训练数据集   
模型存放到./model/mlp/model中   
   
## mlp_model_infer可以用训练好的模型对视频进行处理分类
python mlp_model_infer.py --path dir_path --name mp4_name [--show]   
   
dir_path：数据所在的路径名   
mp4_name: mp4文件名，不包括后缀.mp4   
--show : 处理过程中显示视频图像   
   
## auto_label_mlp用训练好的mlp模型对视频进行自动标注

python auto_label_mlp.py --path dir_path --name mp4_name [--vgg16] \[--show]
   
dir_path：数据所在的路径名   
mp4_name: mp4文件名，不包括后缀.mp4  
--vgg16: 存储处理后的VGG16特征数据   
--show : 处理过程中显示视频图像  


# 机器人避障寻径数据集标注工具

## 使用方法  
python label_mp4_new.py --path dir_path --name mp4_name [--save] [--resume] [--check]   

dir_path：数据所在的路径名   
mp4_name: mp4文件名，不包括后缀.mp4   
--save       : 退出程序时存储数据，若无此项，程序退出前会询问是否存储   
--resume     : 载入以前存储的标注文件，继续进行修改和继续标注   
--check      : 载入以前存储的标注文件，进行修改和继续标注，mp4缓存到内存，mp4所有帧读入后，可以用s,d, k, l键进行浏览修改   
--nav        : 载入先前标注的数据和整个视频，用s, d, k, l键进行浏览修改．没有标注的帧初始化为［３，３，３］   
--iframe NUM : 和--nav一起使用，跳到指定帧进行标注   

标注文件存储在mp4文件同一个目录，名字相同，后缀为.npy 
在浏览修改中，s,k键到前一帧，d, l键到下一帧．（箭头键有问题，故废除）

## 标注数据格式
每帧图像标注为三维数据：［L0, L1, L2]，   
L0表示左测障碍远近，０到７，共８个等级，0表示最近，1表示最远   
L1表示前方障碍远近   
L2表示右方障碍远近   

## 由标注求类别码

根据标注数据，相应的图像帧的类别号码为：   
   c = L0*64 + L1*8 + L2   
共５１２个类别   


## 由类别码求标注
类别 0 <= c < 512   
标注数据为［L0, L1, L2]   
则：   
L0 = c // 64   
L1 = (c mod 64) // 8   
L2 = c mod 8   

# 标注数据处理为VGG16和512类别工具
## 使用方法
python vgg16_label.py --path dir_path --name mp4_name [--show]   
dir_path：数据所在的路径名   
mp4_name: mp4文件名，不包括后缀.mp4   
--show : 处理过程中显示视频图像   
   
vgg16数据和图像512类数据存储在[dir_path]/vgg16_[map_name].npy文件中   
数据格式为：   
{'feats': vgg16_feats, 'labels': vgg16_cls}   