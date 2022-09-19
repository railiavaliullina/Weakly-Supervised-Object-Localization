# Weakly-Supervised-Object-Localization

![image](https://user-images.githubusercontent.com/113569606/190983967-f458e1e6-b131-4399-929f-f5e78ec31d41.png)


## About The Project

1) Implementation of mAP calculation
2) Implementation of process of getting predictions for object localization task:
    
      - boxes prediction based on Class Activarion Maps (CAM), 
      - getting confidences,
      - getting numbers of classes for each picture.
      
3) Getting predictions on COCO2017 validation set
4) Calculating localization accuracy using mAP.


## Getting Started

File to run:

    /trainer/main.py 
    
    or
    
    Weakly-supervised object localization.ipynb
    
    
## Additional Information

Examples of results are in:

    cam_results/
        
 Parameters can be changed in:
 
    configs/config.py
