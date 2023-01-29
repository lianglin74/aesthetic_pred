import torch
import torchvision.transforms as transforms
from lib.init_utils import create_config
from config import cfg as base_config
from  PIL import Image
import torch.nn as nn
from lib.UniCL_multiheads import UniCL_multiheads

def create_config(config_file, opts, print_warning=False):
    cfg = base_config.clone()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    #ensure_config_correctness(cfg, print_warning)
    cfg.freeze()

    return cfg

class ConvertBRG2RGB(object):
   def __call__(self, src_bgr_image):
       src_B, src_G, src_R = src_bgr_image.getchannel('R'), src_bgr_image.getchannel('G'), src_bgr_image.getchannel('B')
       rgb_image = Image.merge('RGB', [src_R, src_G, src_B])

       return rgb_image

def main():
    src_image_file = '/media/lin/Work2/data/ava/ava_tsv/test_images/486521.jpg'
    aesthetics_model_file = '/media/lin/Work2/azure_storage/vigstandard_data/lliang/output/20220512_AVA_DaViT-d5_rating-regress-emd_adamw_b512_lr5e-5_epochs10_cos_wd1e-5_img384_drop-path0.1/model_best.pth.tar'
    model_config_file = '/media/lin/Work2/azure_storage/vigstandard_data/lliang/output/20220512_AVA_DaViT-d5_rating-regress-emd_adamw_b512_lr5e-5_epochs10_cos_wd1e-5_img384_drop-path0.1/config_local.yaml'
    # aesthetics_model_file = '/media/lin/Work2/azure_storage/vigstandard_data/lliang/output/20220512_AVA_DaViT-d3_rating-regress-emd_adamw_b512_lr5e-5_epochs10_cos_wd1e-5_img384/model_best.pth.tar'
    # model_config_file = '/media/lin/Work2/azure_storage/vigstandard_data/lliang/output/20220512_AVA_DaViT-d3_rating-regress-emd_adamw_b512_lr5e-5_epochs10_cos_wd1e-5_img384/config_local.yaml'
    aesthetics_threshold = 7.0

    cfg = create_config(model_config_file, '', print_warning=False)
    checkpoint = torch.load(aesthetics_model_file, map_location='cpu')

    model = UniCL_multiheads(cfg, checkpoint['num_classes'], checkpoint['labelmap'], cfg.MODEL.CLS_RESNET_CONFIG,
                             activation=cfg.MODEL.PRED_ACTIVATION)
    model.load_state_dict(checkpoint['state_dict'])
    #model = model.cuda()
    model.eval()
    del checkpoint

    softmax = nn.Softmax(dim=1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((cfg.INPUT.TRAIN_IMAGE_SIZE), interpolation=Image.BILINEAR),
        transforms.CenterCrop(cfg.INPUT.CROP_SIZE),
        #ConvertBRG2RGB(),
        transforms.ToTensor(),
        normalize,
    ])

    src_image = Image.open(src_image_file)
    input = test_transform(src_image)
    input = input.unsqueeze(0)

    with torch.no_grad():
        aesthetics_prob = model(input)[0]
        aesthetics_prob = softmax(aesthetics_prob)
        print(aesthetics_prob)

    rating_scores = torch.arange(1, 11)
    aesthetics_score = torch.sum(aesthetics_prob * rating_scores, dim = 1)
    print('aesthetics score {}'.format(aesthetics_score))
    if aesthetics_score >= aesthetics_threshold:
        print('aesthetics')
    else:
        print('not aesthetics')

if __name__ == '__main__':
    main()