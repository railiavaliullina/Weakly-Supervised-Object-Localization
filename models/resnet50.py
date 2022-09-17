import torchvision
import torch


def get_model(cfg):
    """
    :param cfg: config
    :return: pretrained on ImageNet resnet-50 model
    """
    print(f'Getting model...')
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.features = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool,
                                            resnet50.layer1,
                                            resnet50.layer2, resnet50.layer3, resnet50.layer4)
    resnet50.sz_features_output = 2048
    # resnet50.features_pooling = torch.nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
    resnet50.features_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    resnet50.fc = torch.nn.Linear(resnet50.sz_features_output, cfg['data']['dataset']['nb_classes'])

    def forward(x):
        x = resnet50.features(x)
        x = resnet50.features_pooling(x)
        bs = x.size(0)
        x = x.view(bs, -1)
        x = resnet50.fc(x)
        return x

    resnet50.forward = forward
    return resnet50
