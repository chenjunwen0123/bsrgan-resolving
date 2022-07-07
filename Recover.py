import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net


class Recover:
    def __init__(self, src):
        self.srcPath = src

        utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
        self.logger = logging.getLogger('blind_sr_log')
        self.picDir = 'static'  # static path
        self.model_name = 'BSRGAN'
        self.save_results = True  # if save result
        self.sf = 4  # scale factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputDir = 'requestPic'  # input Director
        self.outputDir = 'responsePic'  # output Director
        self.L_path = os.path.join(self.picDir, self.inputDir)  # complete input path
        self.E_path = os.path.join(self.picDir, self.outputDir)  # complete output path
        util.mkdir(self.E_path)

    def recover(self):
        # --------------------------------
        # (1) Preinstall
        # --------------------------------
        model_path = os.path.join('model_zoo', self.model_name + '.pth')  # set model path
        torch.cuda.empty_cache()
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=self.sf)  # define network
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False

        model = model.to(self.device)
        torch.cuda.empty_cache()

        # --------------------------------
        # (2) img_L
        # --------------------------------
        img = self.srcPath
        img_name, ext = os.path.splitext(os.path.basename(img))
        self.logger.info('{:<s} --> x{:<d}--> {:<s}'.format(self.model_name, self.sf, img_name + ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(self.device)
        # --------------------------------
        # (3) inference
        # --------------------------------
        img_E = model(img_L)
        # --------------------------------
        # (4) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        if self.save_results:
            savePath = os.path.join(self.E_path, img_name + '.png')
            util.imsave(img_E, savePath)
            return savePath
