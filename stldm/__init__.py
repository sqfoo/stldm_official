from stldm.stldm import model_setup
from stldm.stldm_spatial import model_setup as spatial_setup
from stldm.stldm_mm import model_setup as mm_setup
from stldm.inference import InferenceHub

n2n_setup = {'2D': spatial_setup, '3D': model_setup, 'HKO-H8': mm_setup}