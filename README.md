# Dual Tracker

![](./docs/demo.gif)

inspired by [UAVH](https://openaccess.thecvf.com/content_CVPRW_2019/html/UAVision/Saribas_A_Hybrid_Method_for_Tracking_of_Objects_by_UAVs_CVPRW_2019_paper.html)
## Detector

YOLOv3 with spatial pyramid pooling  
trained on [stanford dataset](https://cvgl.stanford.edu/projects/uav_data/) and [VisDrone](http://aiskyeye.com/).  
spp:  
<img src="./docs/spp.PNG" height="500" width="320"/>  
model is taken from [ultralytics](https://github.com/ultralytics/yolov3)
http://aiskyeye.com/
## Tracker
used opencv trackers


### More References
spp paper https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7005506  
DCFCSR https://arxiv.org/pdf/1611.08461.pdf
