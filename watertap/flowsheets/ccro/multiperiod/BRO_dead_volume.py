import os
import json
import pandas as pd
from collections import defaultdict
from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    Expression,
    Block,
    Param,
    value,
    Var,
    NonNegativeReals,
    assert_optimal_termination,
    Objective,
    units as pyunits,
)
from pyomo.environ import TransformationFactory
from pyomo.network import Arc

from pyomo.util.calc_var_value import calculate_variable_from_constraint

import idaes.core.util.scaling as iscale
from idaes.core.util.initialization import propagate_state
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Product, Feed
from idaes.models.unit_models.mixer import (
    Mixer,
    MomentumMixingType,
    MaterialBalanceType,
)
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

from watertap.costing import (
    WaterTAPCosting,
    PumpType,
    MixerType,
    ROType,
)
from watertap.unit_models.pressure_changer import Pump
from watertap.property_models.NaCl_T_dep_prop_pack import NaClParameterBlock
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
)
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.core.util.initialization import *
from watertap.core.solvers import get_solver
from watertap.flowsheets.ccro.multiperiod.model_state_tool import ModelState

from watertap.unit_models.pseudo_steady_state.dead_volume_0D import DeadVolume0D

# Notes:
# Foundation taken from CCRO_dead_volume.py

# TODO:
# Add costing to each multiperiod block: Fix pump costing,
# Add costing block to the multiperiod block: Capital costing, operating cost from each time step


def build_system(self, time_blk=None):
    """
    Build steady-state model
    """
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = NaClParameterBlock()

    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)

    m.fs.P1 = Pump(property_package=m.fs.properties)
    m.fs.P2 = Pump(property_package=m.fs.properties)

    m.fs.M1 = Mixer(
        property_package=m.fs.properties,
        has_holdup=False,
        num_inlets=2,
        momentum_mixing_type=MomentumMixingType.equality,
    )

    m.fs.RO = ReverseOsmosis1D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
        transformation_scheme="BACKWARD",
        transformation_method="dae.finite_difference",
        finite_elements=10,
        module_type="spiral_wound",
        has_full_reporting=True,
    )

    m.fs.dead_volume = DeadVolume0D(property_package=m.fs.properties)
    # connect unit models
    m.fs.feed_to_P1 = Arc(source=m.fs.feed.outlet, destination=m.fs.P1.inlet)
    m.fs.P1_to_M1 = Arc(source=m.fs.P1.outlet, destination=m.fs.M1.inlet_1)
    m.fs.P2_to_M1 = Arc(source=m.fs.P2.outlet, destination=m.fs.M1.inlet_2)
    m.fs.M1_to_RO = Arc(source=m.fs.M1.outlet, destination=m.fs.RO.inlet)

    m.fs.RO_permeate_to_product = Arc(
        source=m.fs.RO.permeate, destination=m.fs.product.inlet
    )
    m.fs.RO_retentate_to_dead_volume = Arc(
        source=m.fs.RO.retentate, destination=m.fs.dead_volume.inlet
    )

    m.fs.dead_volume_to_P2 = Arc(
        source=m.fs.dead_volume.outlet, destination=m.fs.P2.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.water_recovery = Var(
        initialize=0.5,
        bounds=(0, 0.99),
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        doc="System Water Recovery",
    )

    m.fs.feed_salinity = Var(
        initialize=self.feed_conc,
        bounds=(0, 2000),
        domain=NonNegativeReals,
        units=pyunits.kg / pyunits.m**3,
        doc="Feed salinity",
    )

    m.fs.feed_flow_mass_water = Var(
        initialize=self.feed_flow_mass_water,
        bounds=(0.00001, 1e6),
        domain=NonNegativeReals,
        units=pyunits.kg / pyunits.s,
        doc="Mass flow water",
    )

    m.fs.feed_flow_vol_water = Var(
        initialize=self.feed_flow,
        bounds=(0, None),
        domain=NonNegativeReals,
        units=pyunits.liter / pyunits.min,
        doc="Feed tank, volumetric flow water",
    )

    m.fs.inlet_flow_vol_water = Expression(
        expr=pyunits.convert(
            m.fs.M1.mixed_state[0].flow_vol_phase["Liq"],
            to_units=pyunits.liter / pyunits.minute,
        )
    )

    m.fs.feed.properties[0].flow_vol_phase
    m.fs.feed.properties[0].conc_mass_phase_comp
    m.fs.M1.inlet_1_state[0].flow_vol_phase
    m.fs.M1.inlet_1_state[0].conc_mass_phase_comp
    m.fs.M1.inlet_2_state[0].flow_vol_phase
    m.fs.M1.inlet_2_state[0].conc_mass_phase_comp
    m.fs.M1.mixed_state[0].flow_vol_phase
    m.fs.M1.mixed_state[0].conc_mass_phase_comp

    ### MUST BUILD THIS ON THE MODEL BEFORE PASSING INTO MULTIPERIOD MODEL BLOCK
    m.fs.feed_flow_mass_water_constraint = Constraint(
        expr=m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
        # == m.fs.feed_flow_vol_water * self.rho
        == m.fs.feed_flow_mass_water
    )
    m.fs.feed_flow_salt_constraint = Constraint(
        expr=m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"] * self.rho
        == m.fs.feed_flow_mass_water * self.feed_conc
    )

    add_costing(self, m=None)

    return m

def add_costing(self, m=None):
    """
    Add costing blocks to steady-state model.
    """
    if m is None:
        m = self.m
    m.fs.costing = WaterTAPCosting()
    m.fs.RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.P1.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
    )
    m.fs.P2.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
    )

    m.fs.costing.cost_process()

    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol_phase["Liq"])
    m.fs.costing.add_specific_energy_consumption(
        m.fs.product.properties[0].flow_vol_phase["Liq"], name="SEC"
    )


def create_multiperiod(self,
                       n_time_points=3,
                       ):
    """
    Create MultiPeriod model
    """

    watertap_solver = get_solver()

    self.mp = mp = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=build_system,
        linking_variable_func=get_variable_pairs,
        initialization_func=fix_dof_and_initialize,
        unfix_dof_func=unfix_dof,
        solver=watertap_solver,
        outlvl=logging.WARNING,
    )

    self.flowsheet_options = {
        t: {
            "time_blk": t,
        }
        for t in range(self.n_time_points)
    }

    mp.build_multi_period_model( 
        model_data_kwargs=self.flowsheet_options,
        # unfix_dof_options={"water_recovery": self.water_recovery, "Q_ro": self.inlet_flow_mass_water},
    )
    for t, m in enumerate(mp.get_active_process_blocks()):
        if t == 0:
            self.fix_dof_and_initialize(m=m)
            self.unfix_dof(
                m=m, time_idx=t
            )  # ensure we do not unfix dead volume stuff
            old_m = m
        else:
            self.copy_state_prop_time_period_links(old_m, m)

        results = self.solve(model=m, tee=True)
        assert_optimal_termination(results)
        self.unfix_dof(m=m, time_idx=t)
        old_m = m


def get_variable_pairs(t1, t2):

    return [
        (
            t2.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"],
            t1.fs.dead_volume.dead_volume.properties_out[0].mass_frac_phase_comp[
                "Liq", "NaCl"
            ],
        ),
        (
            t2.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"],
            t1.fs.dead_volume.dead_volume.properties_out[0].dens_mass_phase["Liq"],
        ),
    ]

def copy_state_prop_time_period_links(self, m_old, m_new):
    self.copy_state(m_old, m_new)
    m_new.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"].fix(
        m_old.fs.dead_volume.dead_volume.properties_out[0]
        .mass_frac_phase_comp["Liq", "NaCl"]
        .value
    )
    m_new.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"].fix(
        m_old.fs.dead_volume.dead_volume.properties_out[0]
        .dens_mass_phase["Liq"]
        .value
    )

def copy_state(self, old_model, new_model):
    model_state = ModelState()
    model_state.get_model_state(old_model)
    model_state.set_model_state(new_model)

def copy_inlet_state_for_mixer(self, m):
    for idx, obj in m.fs.M1.inlet_2.flow_mass_phase_comp.items():
        obj.value = m.fs.M1.inlet_1.flow_mass_phase_comp[idx].value * 1

def initialize_system( m=None):
    """
    Initialize steady-state model
    """
    if m is None:
        m = self.m

    # feed is initialized when we setup our fixed operating conditions

    propagate_state(m.fs.feed_to_P1)
    m.fs.P1.outlet.pressure[0].fix(
        m.fs.feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5
    )
    m.fs.P2.outlet.pressure[0].fix(
        m.fs.feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5
    )

    m.fs.P1.initialize()

    propagate_state(m.fs.P1_to_M1)
    self.copy_inlet_state_for_mixer(m)
    m.fs.M1.initialize()
    propagate_state(m.fs.M1_to_RO)

    m.fs.RO.initialize()

    propagate_state(m.fs.RO_permeate_to_product)
    propagate_state(m.fs.RO_retentate_to_dead_volume)
    m.fs.dead_volume.initialize()

    propagate_state(m.fs.dead_volume_to_P2)
    m.fs.P2.initialize()
    m.fs.P2.outlet.pressure[0].unfix()
    propagate_state(m.fs.P2_to_M1)
    m.fs.product.initialize()

def scale_system(self, m=None):
        """
        Scale steady-state model.
        """
        if m is None:
            m = self.m

        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp", 1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp", 1e2, index=("Liq", "NaCl")
        )

        set_scaling_factor(m.fs.P1.control_volume.work, 1e-3)
        set_scaling_factor(m.fs.P2.control_volume.work, 1e-3)
        set_scaling_factor(m.fs.RO.area, 1e-2)

        set_scaling_factor(m.fs.water_recovery, 10)
        set_scaling_factor(m.fs.feed_flow_mass_water, 1)
        set_scaling_factor(m.fs.feed_salinity, 1)

        calculate_scaling_factors(m)


def set_operating_conditions(m=None):
    """
    Set operating conditions as initial conditions.
    """
    if m is None:
        m = m

    m.fs.feed_flow_mass_water.fix(feed_flow_mass_water)
    m.fs.feed_flow_vol_water.fix(feed_flow)
    m.fs.feed_salinity.fix(feed_conc)

    """
    Feed block operating conditions
    """

    m.fs.feed.properties[0].pressure.fix(atmospheric_pressure)
    m.fs.feed.properties[0].temperature.fix(temperature_start)
    # TODO:  These constraints need to be scaled!
    calculate_variable_from_constraint(
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"],
        m.fs.feed_flow_mass_water_constraint,
    )
    calculate_variable_from_constraint(
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"],
        m.fs.feed_flow_salt_constraint,
    )

    """
    Pump 1 operating conditions
    """

    m.fs.P1.efficiency_pump.fix(self.p1_eff)
    m.fs.P1.control_volume.properties_out[0].pressure.fix(self.p1_pressure_start)

    """
    Pump 2 operating conditions
    """

    m.fs.P2.efficiency_pump.fix(self.p2_eff)
    # m.fs.P2.control_volume.properties_out[0].pressure.set_value(self.p2_pressure_start)

    """
    RO operating conditions
    """

    m.fs.RO.permeate.pressure[0].fix(atmospheric_pressure)
    m.fs.RO.A_comp.fix(A_comp)
    m.fs.RO.B_comp.fix(B_comp)
    m.fs.RO.area.fix(membrane_area)
    m.fs.RO.length.fix(membrane_length)
    m.fs.RO.feed_side.channel_height.fix(channel_height)
    m.fs.RO.feed_side.spacer_porosity.fix(spacer_porosity)

    m.fs.RO.feed_side.K.setlb(1e-6)
    m.fs.RO.feed_side.friction_factor_darcy.setub(200)
    m.fs.RO.flux_mass_phase_comp.setub(1)
    m.fs.RO.feed_side.cp_modulus.setub(50)
    m.fs.RO.feed_side.cp_modulus.setlb(0.1)
    m.fs.RO.deltaP.setlb(None)
    for e in m.fs.RO.permeate_side:
        if e[-1] != 0:
            m.fs.RO.permeate_side[e].pressure_osm_phase["Liq"].setlb(200)
            m.fs.RO.permeate_side[e].molality_phase_comp["Liq", "NaCl"].setlb(1e-8)
    # assume there is no change in volume
    m.fs.dead_volume.volume.fix(dead_volume)
    m.fs.dead_volume.delta_state.volume.fix(dead_volume)
    m.fs.dead_volume.accumulation_time.fix(accumulation_time)

    # initialize state to assumed initial condition, use concentration of feed
    # and density of feed
    # Solve feed block first

    m.fs.feed.properties[0].pressure_osm_phase["Liq"]
    m.fs.feed.properties[0].flow_vol_phase["Liq"].fix(reject_flow)
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].fix(
        reject_conc_start
    )
    solver = get_solver()
    solver.solve(m.fs.feed)
    m.fs.feed.properties[0].flow_vol_phase["Liq"].unfix()
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].unfix()

    # I found fixing mass fraction and density is easiest way to get initial state
    # we will also use these as connection points between current and future state.

    m.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"].fix(
        m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "NaCl"].value
    )
    m.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"].fix(
        m.fs.feed.properties[0].dens_mass_phase["Liq"].value
    )

    # initialize feed to desired flow and conc
    m.fs.feed.properties[0].flow_vol_phase["Liq"].fix(feed_flow)
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].fix(feed_conc)
    solver = get_solver()
    solver.solve(m.fs.feed)
    m.fs.feed.properties[0].flow_vol_phase["Liq"].unfix()
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].unfix()
    print("DOF =", degrees_of_freedom(m))
    print("DOF FEED =", degrees_of_freedom(m.fs.feed))
    print("DOF PUMP 1 =", degrees_of_freedom(m.fs.P1))
    print("DOF PUMP 2 =", degrees_of_freedom(m.fs.P2))
    print("DOF MIXER =", degrees_of_freedom(m.fs.M1))
    print("DOF RO =", degrees_of_freedom(m.fs.RO))
    print("DOF Dead Volume =", degrees_of_freedom(m.fs.dead_volume))
    assert_no_degrees_of_freedom(m)
    scale_system(m)

def fix_dof_and_initialize(m=None):
    """
    Fix DOF for MP model and initialize steady-state models.
    """
    if m is None:
        m = m

    set_operating_conditions(m=m)
    initialize_system(m=m)


def unfix_dof(self, time_idx=0, m=None):
    """
    Unfix linking variables in MP model
    """
    if m is None:
        m = self.m
    # m.fs.feed_flow_vol_water.unfix()
    # m.fs.feed_flow_mass_water.unfix()
    # m.fs.RO.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
    #     self.inlet_flow_mass_water
    # )
    # m.fs.RO.recovery_vol_phase[0, "Liq"].fix(self.water_recovery)

    # m.fs.recov_constr = Constraint(expr=m.fs.RO.recovery_vol_phase[0, "Liq"] >= self.water_recovery)
    # m.fs.eq_flow_eq = Constraint(
    #     expr=m.fs.RO.mixed_permeate[0].flow_vol_phase["Liq"]
    #     == m.fs.feed.properties[0].flow_vol_phase["Liq"]
    # )
    m.fs.P1.control_volume.properties_out[0].pressure.unfix()
    m.fs.P2.control_volume.properties_out[0].pressure.unfix()

    m.fs.P2.control_volume.properties_out[0].flow_vol_phase["Liq"].fix(
        self.reject_flow
    )

    if time_idx > 0:
        # m.fs.RO.recovery_vol_phase[0, "Liq"].fix(self.water_recovery)
        # m.fs.P1.control_volume.properties_out[0].pressure.unfix()
        # m.fs.P2.control_volume.properties_out[0].pressure.unfix()
        m.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"].unfix()
        m.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"].unfix()


def solve(self, model=None, solver=None, tee=False, raise_on_failure=True):
    # ---solving---
    if solver is None:
        solver = get_solver()

    if model is None:
        model = self.mp

    print("\n--------- SOLVING ---------\n")
    print(f"Degrees of Freedom: {degrees_of_freedom(model)}")
    results = solver.solve(model, tee=tee)
    if check_optimal_termination(results):
        print("\n--------- OPTIMAL SOLVE!!! ---------\n")
        return results
    msg = "The current configuration is infeasible. Please adjust the decision variables."
    if raise_on_failure:
        print("\n--------- INFEASIBLE SOLVE!!! ---------\n")

        print("\n--------- CLOSE TO BOUNDS ---------\n")
        print_close_to_bounds(model)

        print("\n--------- INFEASIBLE BOUNDS ---------\n")
        print_infeasible_bounds(model)

        print("\n--------- INFEASIBLE CONSTRAINTS ---------\n")
        print_infeasible_constraints(model)

        raise RuntimeError(msg)
    else:
        print(msg)
        # debug(model, solver=solver, automate_rescale=False, resolve=False)
        # check_jac(model)
        assert False


if __name__ == "__main__":
