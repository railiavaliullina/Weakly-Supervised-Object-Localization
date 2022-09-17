cfg = {
    # parameters for dataset and dataloader
    'data':
        {
            'dataset':
                {
                    'root_path': 'D:/datasets/homeworks/cv segmentation/',
                    'nb_train_images': 562331,
                    'nb_val_images': 36334,
                    'nb_classes': 80
                },
            'dataloader':
                {
                    'batch_size': 16
                }


        },

    # parameters for setting up training parameters
    'train':
        {
            'optimizer':
                {
                    'lr': 1e-3,
                    'weight_decay': 1e-4
                },
            'epochs': 100

        },

    # parameters for model evaluation
    "eval":
        {
            'evaluate_on_train_data': False,
            'evaluate_before_training': False,
        },

    # parameters for logging training process, saving/restoring model
    "logging":
        {
            'log_metrics': True,
            'experiment_name': 'coco_classification',
            'checkpoints_dir': 'checkpoints/',
            'save_model': True,
            'load_model': False,
            'epoch_to_load': 20,
            'save_frequency': 1,
        },

    # parameters to debug training and check if everything is ok
    "debug":
        {
            # to check batches before training
            "save_batch":
                {
                    "enable": False,
                    "nrof_batches_to_save": 5,
                    "path_to_save": 'batches_images/',
                },
            "overfit_on_batch":
                {
                    "enable": False,
                    "nb_iters": 1000,
                }
        },

    # parameters for plotting CAMs
    "CAM":
        {
            # 'checkpoints_dir': 'D:/datasets/homeworks/cv weakly-supervised object localization/kaggle checkpoints/finetune_fc/',
            # 'epoch_to_load': 0,
            'checkpoints_dir': 'D:/datasets/homeworks/cv weakly-supervised object localization/kaggle checkpoints/last/',
            'epoch_to_load': 3,
            'path_to_save': 'D:/datasets/homeworks/cv weakly-supervised object localization/CAM/'
        },
    'localization':
        {
            'pickle_path': 'predictions_3_correct_softmax.pickle',  # 0 - 1.03, 3 - 1.14, 2 - 1.08
            'save_bboxes': True,
        }
}
