from models import resnet
from models import gan_models


def choose_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet.resnet18(**kwargs)
    else:
        raise ValueError('Wrong model name.')

def choose_g_model(model_name, **kwargs):
    if model_name == 'CGeneratorA':
        return gan_models.CGeneratorA(**kwargs)
    else:
        raise ValueError('Wrong model name.')
