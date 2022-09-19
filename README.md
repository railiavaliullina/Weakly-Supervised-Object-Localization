# Weakly-Supervised-Object-Localization

### Results example:

- CAM visualization,
- Predicted class: giraffe, correct: giraffe.
![image](https://user-images.githubusercontent.com/113569606/191008790-064901e9-524d-415a-8f39-d8c6d48a1266.png)


## About The Project

1) Implementation of mAP calculation
2) Implementation of process of getting predictions for object localization task:
    
      - getting bounding boxes based on Class Activarion Maps (CAM), 
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
