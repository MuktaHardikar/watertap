from pyomo.environ import (
    ConcreteModel,
    Param,
    check_optimal_termination,
    value,
    assert_optimal_termination,
    units as pyunits,
    value,
    TransformationFactory,
)

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import MixingType, MomentumMixingType, Mixer, Separator, Product, Feed

#TODO:
#1. Create case study yaml file for WRD treatment train

def build_wrd_treatment_train():
    pass

def set_wrd_treatment_train_operating_conditions(m):
    pass

def add_connections(m):
    pass

def initialize_wrd_treatment_train(m, solver=None, outlvl=1, optarg=None):
    pass

def set_wrd_treatment_train_scaling(m):
    pass

def solve_wrd_treatment_train(m, solver=None, tee=False):
    pass
