from .unet_model import model as model
from .unet_model import config as config
#from .train import train_net as train
from .train import train_net_loss as train_loss
from .train_seg import train_net
from .test_loc import test_net