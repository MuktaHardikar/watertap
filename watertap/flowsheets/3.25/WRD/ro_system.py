from xml.parsers.expat import model
import matplotlib.pyplot as plt
from pyomo.environ import (
    ConcreteModel,
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

from pyomo.network import Arc
import idaes.core.util.scaling as iscale

from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice

from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    PressureChangeType,
    MassTransferCoefficient,
    ConcentrationPolarizationType,
)
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.property_models.NaCl_T_dep_prop_pack import NaClParameterBlock
from watertap.core.solvers import get_solver

from idaes.core.util.scaling import (
    constraint_scaling_transform,
    calculate_scaling_factors,
    set_scaling_factor,
)

import yaml
import os

#TODO:
# Flowrate at each stage needs to aggregated and recalculated

solver = get_solver()

def load_config(config):
    with open(config, "r") as file:
        return yaml.safe_load(file)

def get_config_value(config, key, section, subsection=None,):
    """
    Get a value from the configuration file.
    """

    if section in config:
        if subsection:
            if subsection in config[section]:
                if key in config[section][subsection]:
                    if isinstance(config[section][subsection][key], dict) and "value" in config[section][subsection][key] and "units" in config[section][subsection][key]:
                        return config[section][subsection][key]["value"] * getattr(pyunits, config[section][subsection][key]["units"])
                    return config[section][subsection][key]
                else:
                    raise KeyError(f"Key '{key}' not found in subsection '{subsection}' of section '{section}' of the configuration.")
            else:
                raise KeyError(f"Section '{section}' or subsection '{subsection}' not found in the configuration.")
        else:
            if key in config[section]:
                if isinstance(config[section][key], dict) and "value" in config[section][key] and "units" in config[section][key]:
                    return config[section][key]["value"] * getattr(pyunits, config[section][key]["units"])
                return config[section][key]
            else:
                raise KeyError(f"Key '{key}' not found in section '{section}' of the configuration.")
    else:
        raise KeyError(f"Section '{section}' not found in the configuration.")
   
def build_wrd_ro_system():
    '''
    Build reverse osmosis system for WRD
    '''
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = NaClParameterBlock()

    # Feed stream to first pump and system permeate
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.permeate = Product(property_package=m.fs.properties)

    cwd = os.path.dirname(os.path.abspath(__file__))
    config = cwd + "/wrd_inputs.yaml"
    m.fs.config_data = load_config(config)

    m = add_units(m)
    print("Degrees of freedom after adding units:", degrees_of_freedom(m))

    m = build_inlet_stream(m)
    print("Degrees of freedom after setting inlet stream:", degrees_of_freedom(m))
    m = set_operation_conditions(m)
    print("Degrees of freedom after setting operation conditions:", degrees_of_freedom(m))

    m = add_connections(m)
    print("Degrees of freedom before initialization:", degrees_of_freedom(m))

    add_scaling(m)
         
    m = initialize_units(m)

    return m

def add_units(m):

    # Feed pump to first stage RO
    m.fs.pump1 = Pump(property_package=m.fs.properties)

    # Feed pump to second stage RO
    m.fs.pump2 = Pump(property_package=m.fs.properties)

    # Feed pump to third stage RO
    # m.fs.pump3 = Pump(property_package=m.fs.properties)

    # Three stages of reverse osmosis
    for i in range(1, 3):
        setattr(
            m.fs,
            f"ro_stage_{i}",
            ReverseOsmosis1D(
                    property_package=m.fs.properties,
                    has_pressure_change=True,
                    pressure_change_type=PressureChangeType.calculated,
                    mass_transfer_coefficient=MassTransferCoefficient.calculated,
                    concentration_polarization_type=ConcentrationPolarizationType.calculated,
                    transformation_scheme="BACKWARD",
                    transformation_method="dae.finite_difference",
                    module_type="spiral_wound",
                    finite_elements=7, 
                    has_full_reporting=True,
            )
        )
    
    # Add permeate mixer
    # m.fs.permeate_mixer = Mixer(
    #     property_package=m.fs.properties,
    #     inlet_list=["ro_stage_1_permeate", "ro_stage_2_permeate", "ro_stage_3_permeate"],
    #     energy_mixing_type=MixingType.extensive,
    #     momentum_mixing_type=MomentumMixingType.minimize,
    # )
    
    return m

def build_inlet_stream(m):
    '''Build the inlet stream for the RO system'''
    # Check if permeate flowrate could be used instead

    # The feed stream is divided by the number of trains and vessels in stage 1
    equal_division_factor = get_config_value(m.fs.config_data,"number_of_trains", "reverse_osmosis_1d")

    feed_mass_flow_water = get_config_value(m.fs.config_data,"feed_flow_water", "feed_stream")\
                            *get_config_value(m.fs.config_data,"feed_density_water", "feed_stream")/ equal_division_factor
    
    feed_mass_flow_salt = get_config_value(m.fs.config_data,"feed_conductivity", "feed_stream") \
                            *get_config_value(m.fs.config_data,"feed_conductivity_conversion", "feed_stream")\
                            *get_config_value(m.fs.config_data,"feed_flow_water", "feed_stream")/ equal_division_factor
    
    feed_temperature = get_config_value(m.fs.config_data,"feed_temperature", "feed_stream")
    feed_pressure = get_config_value(m.fs.config_data,"feed_pressure", "feed_stream")

    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(feed_mass_flow_water)
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"].fix(feed_mass_flow_salt)
    m.fs.feed.properties[0].temperature.fix(feed_temperature)
    m.fs.feed.properties[0].pressure.fix(feed_pressure)

    return m

def set_operation_conditions(m):
    '''
    Set the operation conditions for the RO system
    '''
    # Set pump operating conditions
    for i in range(1, 3):
        pump = getattr(m.fs, f"pump{i}")
        
        pump.control_volume.properties_out[0].pressure.fix(
            get_config_value(m.fs.config_data, "pump_outlet_pressure", "pumps", f"pump_{i}")
        )

        pump.efficiency_pump.fix(get_config_value(m.fs.config_data, "pump_efficiency", "pumps", f"pump_{i}"))


    # Set RO configuration for each stage
    for i in range(1, 3):
        ro_stage = getattr(m.fs, f"ro_stage_{i}")

        ro_stage.A_comp.fix(get_config_value(m.fs.config_data, "A_comp", "reverse_osmosis_1d", f"stage_{i}"))
        ro_stage.B_comp.fix(get_config_value(m.fs.config_data, "B_comp", "reverse_osmosis_1d", f"stage_{i}"))
        
        ro_stage.feed_side.channel_height.fix(get_config_value(m.fs.config_data, "channel_height", "reverse_osmosis_1d", f"stage_{i}"))
        ro_stage.feed_side.spacer_porosity.fix(get_config_value(m.fs.config_data, "spacer_porosity", "reverse_osmosis_1d", f"stage_{i}"))

        # ro_stage.feed_side.length.fix(
        #     get_config_value(m.fs.config_data, "number_of_elements_per_vessel", "reverse_osmosis_1d", f"stage_{i}")*
        #     get_config_value(m.fs.config_data, "element_length", "reverse_osmosis_1d", f"stage_{i}")
        # )
        
        ro_stage.area.fix(
            get_config_value(m.fs.config_data, "element_membrane_area", "reverse_osmosis_1d", f"stage_{i}")*
            get_config_value(m.fs.config_data, "number_of_vessels", "reverse_osmosis_1d", f"stage_{i}")*
            get_config_value(m.fs.config_data, "number_of_elements_per_vessel", "reverse_osmosis_1d", f"stage_{i}")
        )

        # ro_stage.permeate.pressure[0].fix(101325)
        pressure_loss = -9 * pyunits.psi
        ro_stage.deltaP.fix(pressure_loss)
        
        ro_stage.recovery_mass_phase_comp[0,"Liq", "H2O"].fix(get_config_value(m.fs.config_data, "water_recovery_mass_phase", "reverse_osmosis_1d", f"stage_{i}"))
        
    return m


def relax_bounds_for_low_salinity_waters(blk):
    blk.feed_side.cp_modulus.setub(5)
    for e in blk.feed_side.K:
        blk.feed_side.K[e].setub(0.01)
        # blk.feed_side.K[e].setlb(1e-7)

    for e in blk.feed_side.cp_modulus:
        blk.feed_side.cp_modulus[e].setlb(1e-5)

    for e in blk.recovery_mass_phase_comp:
        if e[-1] == "NaCl":
            blk.recovery_mass_phase_comp[e].setlb(1e-9)
            blk.recovery_mass_phase_comp[e].setub(1e-1)

    for e in blk.flux_mass_phase_comp:
        if e[-1] == "NaCl":
            blk.flux_mass_phase_comp[e].setlb(1e-9)
            blk.flux_mass_phase_comp[e].setub(1e-1)

    for e in blk.recovery_mass_phase_comp:
        if e[-1] == "H2O":
            blk.recovery_mass_phase_comp[e].setlb(1e-4)
            blk.recovery_mass_phase_comp[e].setub(0.999)

    for e in blk.flux_mass_phase_comp:
        if e[-1] == "H2O":
            blk.flux_mass_phase_comp[e].setlb(1e-5)
            blk.flux_mass_phase_comp[e].setub(0.999)


def add_connections(m):
    '''
    Add connections between the units in the RO system
    '''

    # Connect feed to first pump
    m.fs.feed_to_pump1 = Arc(source=m.fs.feed.outlet, destination=m.fs.pump1.inlet)

    # Connect first pump to first RO stage
    m.fs.pump1_to_ro_stage_1 = Arc(source=m.fs.pump1.outlet, destination=m.fs.ro_stage_1.inlet)

    # Connect first RO stage to second pump
    # m.fs.ro_stage_1_to_pump2 = Arc(source=m.fs.ro_stage_1.retentate, destination=m.fs.pump2.inlet)

    # Connect second pump to second RO stage
    # m.fs.pump2_to_ro_stage_2 = Arc(source=m.fs.pump2.outlet, destination=m.fs.ro_stage_2.inlet)

    # Connect second RO stage to third pump
    # m.fs.ro_stage_2_to_pump3 = Arc(source=m.fs.ro_stage_2.retentate, destination=m.fs.pump3.inlet)

    # Connect third pump to third RO stage
    # m.fs.pump3_to_ro_stage_3 = Arc(source=m.fs.pump3.outlet, destination=m.fs.ro_stage_3.inlet)

    # Connect permeate from first and second stages to permeate mixer
    # m.fs.ro_stage_1_to_permeate_mixer = Arc(
    #     source=m.fs.ro_stage_1.permeate,
    #     destination=m.fs.permeate_mixer.ro_stage_1_permeate,
    # )
    
    # m.fs.ro_stage_2_to_permeate_mixer = Arc(
    #     source=m.fs.ro_stage_2.permeate,
    #     destination=m.fs.permeate_mixer.ro_stage_2_permeate,
    # )

    # Connect third RO stage to permeate mixer
    # m.fs.ro_stage_3_to_permeate_mixer = Arc(
    #     source=m.fs.ro_stage_3.permeate,
    #     destination=m.fs.permeate_mixer.ro_stage_3_permeate,
    # )

    # Connect permeate mixer to permeate product stream
    # m.fs.permeate_mixer_to_permeate = Arc(
    #     source=m.fs.permeate_mixer.outlet,
    #     destination=m.fs.permeate.inlet
    # )

    TransformationFactory("network.expand_arcs").apply_to(m)

    return m

def add_scaling(m):
    '''
    Add scaling to the units in the RO system
    '''
    calculate_scaling_factors(m.fs.feed.properties[0])
    calculate_scaling_factors(m.fs.permeate.properties[0])

    for i in range(1, 2):
        pump = getattr(m.fs, f"pump{i}")
        calculate_scaling_factors(pump)

        ro_stage = getattr(m.fs, f"ro_stage_{i}")
        calculate_scaling_factors(ro_stage)

        # Calculate RO scaling factors
        set_scaling_factor(ro_stage.feed_side.spacer_porosity, 1e-1)
        set_scaling_factor(ro_stage.feed_side.channel_height, 1e-5)
        constraint_scaling_transform(ro_stage.feed_side.eq_dh, 1e-6)
        constraint_scaling_transform(ro_stage.feed_side.eq_N_Re[0.0,1.0], 1e-6)

        set_scaling_factor(ro_stage.area, 1e3)

    return m

def initialize_units(m):
    '''
    Initialize the units in the RO system
    '''
    calculate_scaling_factors(m)

    # Initialize feed stream
    m.fs.feed.initialize()
    print("Degrees of freedom after feed initialization:", degrees_of_freedom(m))

    # Propagate feed state to pump
    propagate_state(m.fs.feed_to_pump1)

    # Initialize pumps and RO
    for i in range(1, 2):
        pump = getattr(m.fs, f"pump{i}")
        pump.initialize()

        # Propagate state from pump to RO stage
        propagate_state(getattr(m.fs, f"pump{i}_to_ro_stage_{i}"))

        ro_stage = getattr(m.fs, f"ro_stage_{i}")

        print("Degrees of freedom after pump initialization:", degrees_of_freedom(m))

        relax_bounds_for_low_salinity_waters(ro_stage)

        try:
            ro_stage.initialize()
        except:
            print_infeasible_constraints(ro_stage)
            print("Degrees of freedom after RO initialization:", degrees_of_freedom(m))

        # Propagate state from RO stage to permeate mixer
        # propagate_state(getattr(m.fs, f"ro_stage_{i}_to_permeate_mixer"))

    # Initialize permeate mixer
    # m.fs.permeate_mixer.initialize()

    # propagate_state(m.fs.permeate_mixer_to_permeate)
    # m.fs.permeate.initialize()
    
    return m



if __name__ == "__main__":

    m = build_wrd_ro_system()

    print("Degrees of freedom:", degrees_of_freedom(m))


    try:
        results = solver.solve(m, tee=False)
    except:
        print_infeasible_constraints(m)
    