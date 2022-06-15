import argparse
import json


def modify_command_options(opts):


    return opts


def get_argparser():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--model", type=str, default='resnet34')    #, choices=['resnet', 'vgg11', 'EfficientNet', 'ViT', 'resnet34']
    parser.add_argument("--epochs", type=int, default=30, help="epoch number (default: 30)")
    parser.add_argument("--batch_size", type=int, default=256, help='batch size (default: 128)')
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.007)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="check point model"
    )
    parser.add_argument(
        "--model_name", type=str, default='clean_model', help="checkpoint modelname"
    )
    parser.add_argument(
        "--beta", type=int, default='1', help="beta value"
    )
    # parser.add_argument("--diff_model", type=str, default='diff_model4')
    # parser.add_argument("--diff_result", type=str, default='diff_result4')
    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.

    # Train Options
#     parser.add_argument("--epochs", type=int, default=30, help="epoch number (default: 30)")
#     parser.add_argument("--batch_size", type=int, default=4, help='batch size (default: 4)')
#     parser.add_argument("--crop_size", type=int, default=512, help="crop size (default: 513)")

    


    return parser
