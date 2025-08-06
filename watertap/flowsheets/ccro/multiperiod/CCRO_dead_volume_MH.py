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
from idaes.models.unit_models import Product, Feed, StateJunction
from idaes.models.unit_models.mixer import (
    Mixer,
    MomentumMixingType,
    MaterialBalanceType,
)
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

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

#TODO:
# Track feed inlet flow and permeate production

atmospheric_pressure = 101325 * pyunits.Pa
solver = get_solver()


def get_variable_pairs(t1, t2):
    """
    Get variable pairs for connecting two time periods.
    1. dead_volume mass fraction to delta_state
    2. dead_volume density to delta_state
    3. product flowrate to feed flowrate - This is not valid for flushing 
    """

    return [

        (
            t2.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"],
            t1.fs.dead_volume.dead_volume.properties_out[0].mass_frac_phase_comp[
                "Liq", "NaCl"
            ],
        ),

        # (
        #     t2.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "H2O"],
        #     t1.fs.dead_volume.dead_volume.properties_out[0].mass_frac_phase_comp[
        #         "Liq", "H2O"
        #     ],
        # ),

        (
            t2.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"],
            t1.fs.dead_volume.dead_volume.properties_out[0].dens_mass_phase["Liq"],
        ),

        (
            t2.fs.dead_volume.dead_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
            t1.fs.RO.retentate.flow_mass_phase_comp[0, "Liq", "H2O"]
        ),

        (
            t2.fs.dead_volume.dead_volume.properties_in[0].flow_mass_phase_comp["Liq", "NaCl"],
            t1.fs.RO.retentate.flow_mass_phase_comp[0, "Liq", "NaCl"]
        )

    ]


class CCRO_dead_volume_flushing:

    """
    Class to create a CCRO flowsheet with dead volume and flushing.
    """

    def __init__(
        self,
        n_time_points = 5,
        raw_feed_conc= None,
        raw_feed_flowrate = None,
        recycle_flowrate = None,
        recycle_conc_start = None,
        temperature_start = None,
        p1_pressure_start = None,
        p2_pressure_start = None,
        p1_eff = 0.8,
        p2_eff = 0.8,
        A_comp = 4.2e-12,
        B_comp = 3.5e-8,
        membrane_area = 7.2,  # m2
        membrane_length = 0.9626,  # m
        channel_height = 1e-3,
        spacer_porosity = 0.97,
        dead_volume = 0.010942,
        accumulation_time = 120,
    ):
        
        self.n_time_points = n_time_points
        self.raw_feed_conc = raw_feed_conc * pyunits.g/pyunits.L
        self.raw_feed_flowrate = raw_feed_flowrate * pyunits.L/pyunits.min
        self.recycle_conc_start = recycle_conc_start * pyunits.g/pyunits.L
        self.rho = 1005 * pyunits.kg / pyunits.m**3
        self.recycle_flowrate = recycle_flowrate * pyunits.L/pyunits.min
        self.temperature_start = (temperature_start + 273.15) * pyunits.degK
        self.p1_pressure_start = p1_pressure_start * pyunits.Pa
        if p2_pressure_start is not None:
            self.p2_pressure_start = p2_pressure_start * pyunits.psi
        else:
            self.p2_pressure_start = self.p1_pressure_start
        self.p1_eff = p1_eff
        self.p2_eff = p2_eff
        self.A_comp = A_comp
        self.B_comp = B_comp
        self.membrane_area = membrane_area * pyunits.m**2
        self.membrane_length = membrane_length * pyunits.m
        self.channel_height = channel_height * pyunits.m
        self.spacer_porosity = spacer_porosity
        self.dead_volume = dead_volume * pyunits.m**3
        self.accumulation_time = accumulation_time * pyunits.s

        # Calculate _flow_mass for raw feed and recycle
        self.raw_feed_flow_mass_water = pyunits.convert(
            self.rho * self.raw_feed_flowrate, to_units=pyunits.kg / pyunits.s
        )
        self.raw_feed_flow_mass_salt = pyunits.convert(
            self.raw_feed_conc * self.raw_feed_flowrate, to_units=pyunits.kg / pyunits.s
        )

        self.recycle_flowrate = pyunits.convert(
            self.recycle_flowrate, to_units=pyunits.m**3 / pyunits.s
        )

        self.recycle_conc_start = pyunits.convert(
            self.recycle_conc_start, to_units=pyunits.kg/pyunits.m**3
        )


        self.recycle_flow_mass_water = pyunits.convert(
            self.rho * self.recycle_flowrate, to_units=pyunits.kg / pyunits.s
        )

        if self.recycle_conc_start is not None:
            # Recycle concentration should be a function of cycle recovery
            self.recycle_flow_mass_salt = pyunits.convert(
                self.recycle_conc_start * self.recycle_flowrate, to_units=pyunits.kg / pyunits.s
            )
        

    def build_system(self,time_blk=None):

        """
        Build the CCRO steady state flowsheet
        """

        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = NaClParameterBlock()

        # Raw feed to the system
        m.fs.raw_feed = Feed(property_package=m.fs.properties)
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

        # Add connections
        m.fs.raw_feed_to_P1 = Arc(source=m.fs.raw_feed.outlet, destination=m.fs.P1.inlet)

        m.fs.dead_volume_to_P2 = Arc(
            source=m.fs.dead_volume.outlet, destination=m.fs.P2.inlet
        )

        m.fs.P1_to_M1 = Arc(source=m.fs.P1.outlet, destination=m.fs.M1.inlet_1)
        m.fs.P2_to_M1 = Arc(source=m.fs.P2.outlet, destination=m.fs.M1.inlet_2)
        
        m.fs.M1_to_RO = Arc(source=m.fs.M1.outlet, destination=m.fs.RO.inlet)

        m.fs.RO_permeate_to_product = Arc(
            source=m.fs.RO.permeate, destination=m.fs.product.inlet
        )

        TransformationFactory("network.expand_arcs").apply_to(m)

        m.fs.single_pass_water_recovery = Var(
            initialize=0.5,
            bounds=(0, 0.99),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Water recovery after single pass through RO or accumulation time",
        )

        
        return m
    
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
        set_scaling_factor(m.fs.RO.area, 1e-1)

        set_scaling_factor(m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1)
        set_scaling_factor(m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"], 1e-2)

        calculate_scaling_factors(m)


    def set_operating_conditions(self, m):

        if m is None:
            m = self.m

        """
        Feed block operating conditions
        """
        m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(self.raw_feed_flow_mass_water)
        m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"].fix(self.raw_feed_flow_mass_salt)
        m.fs.raw_feed.properties[0].pressure.fix(atmospheric_pressure)
        m.fs.raw_feed.properties[0].temperature.fix(self.temperature_start)

        """
        Dead Volume operating conditions
        """
        # Fixed volume
        m.fs.dead_volume.volume.fix(self.dead_volume)
        m.fs.dead_volume.delta_state.volume.fix(self.dead_volume)

        # Fixed for accumulation time for initialization
        m.fs.dead_volume.accumulation_time.fix(self.accumulation_time)

        # m.fs.dead_volume.dead_volume.mass_frac_phase_comp[0, "Liq", "H2O"].fix(
        #     self.recycle_flow_mass_water/(self.recycle_flow_mass_water + self.recycle_flow_mass_salt)
        # )

        # m.fs.dead_volume.dead_volume.mass_frac_phase_comp[0, "Liq", "NaCl"].fix(
        #     self.recycle_flow_mass_salt/(self.recycle_flow_mass_water + self.recycle_flow_mass_salt)
        # )

        # Fixing the flow rate of the dead volume outlet
        m.fs.dead_volume.dead_volume.properties_out.calculate_state(
            var_args={
                ("flow_vol_phase","Liq"):self.recycle_flowrate,
                # ("conc_mass_phase_comp", ("Liq", "NaCl")): self.recycle_conc_start,
                ("flow_mass_phase_comp", ("Liq", "NaCl")): self.recycle_flow_mass_salt,
                ("temperature", None): self.temperature_start
            },
            hold_state=True

        )
        
        m.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"].fix(0.0115)
        m.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"].fix(1004)
        

        """
        Pump 1 operating conditions
        """

        m.fs.P1.efficiency_pump.fix(self.p1_eff)

        """
        Pump 2 operating conditions
        """

        m.fs.P2.efficiency_pump.fix(self.p2_eff)
       
        """
        RO operating conditions
        """
        # Set RO configuration parameters
        m.fs.RO.A_comp.fix(self.A_comp)
        m.fs.RO.B_comp.fix(self.B_comp)
        m.fs.RO.area.fix(self.membrane_area)
        m.fs.RO.length.fix(self.membrane_length)
        m.fs.RO.feed_side.channel_height.fix(self.channel_height)
        m.fs.RO.feed_side.spacer_porosity.fix(self.spacer_porosity)

        m.fs.RO.permeate.pressure[0].fix(atmospheric_pressure)

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


        m.fs.M1.mixed_state[0].flow_vol_phase
        m.fs.M1.mixed_state[0].conc_mass_phase_comp

        # Single pass water recovery constraint
        single_pass_water_recovery_constraint = Constraint(
            expr = m.fs.RO.permeate.flow_mass_phase_comp[0,"Liq", "H2O"]/self.rho
                == m.fs.single_pass_water_recovery
                * m.fs.M1.mixed_state[0].flow_vol_phase["Liq"]
            )
        
        self.scale_system(m)

        return m

    def initialize_system(self, m):
        """
        Initialize the model by fixing the values of certain variables.
        """

        m.fs.raw_feed.initialize()

        propagate_state(m.fs.raw_feed_to_P1)
        m.fs.P1.initialize()

        m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"]
        # m.fs.P1.outlet.pressure[0].fix(
        #     m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5
        # )

        m.fs.dead_volume.initialize()
        # m.fs.dead_volume.dead_volume.properties_out[0].pressure.unfix()

        propagate_state(m.fs.dead_volume_to_P2)

        # m.fs.P2.outlet.pressure[0].fix(
        #     m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5
        # )

        m.fs.P2.initialize()

        propagate_state(m.fs.P1_to_M1)
        propagate_state(m.fs.P2_to_M1)
        
        m.fs.M1.initialize()
        m.fs.M1.mixed_state[0].pressure.fix(m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5)

        propagate_state(m.fs.M1_to_RO)
        m.fs.RO.initialize()

        propagate_state(m.fs.RO_permeate_to_product)

        m.fs.product.initialize()

        return m
    
    
    def create_multiperiod(self):
        """
        Create MultiPeriod model
        """

        watertap_solver = get_solver()

        self.mp = mp = MultiPeriodModel(
            n_time_points=self.n_time_points,
            process_model_func=self.build_system,
            linking_variable_func=get_variable_pairs,
            initialization_func=self.fix_dof_and_initialize,
            unfix_dof_func=self.unfix_dof,
            solver=watertap_solver,
            outlvl=logging.WARNING,
        )

        self.flowsheet_options = {
            t: {
                "time_blk": t,
            }
            for t in range(self.n_time_points)
        }
        
        # Build instances of the process model for each time period
        mp.build_multi_period_model(
            model_data_kwargs=self.flowsheet_options,
            # unfix_dof_options={"water_recovery": self.water_recovery, "Q_ro": self.inlet_flow_mass_water},
        )

        solver = get_solver()
        # Initialize all time periods. Initialize the delta state using dead volume outlet properties
        for t, m in enumerate(mp.get_active_process_blocks()):

            self.fix_dof_and_initialize(m=m)
            print("Solving for time period", t)
            try:
               results = self.solve(model=m, tee=False)
            except:
                print("Infeasible solution for time period", t)
                print_infeasible_constraints(m)
                print('\n')
            
            if t == 0:    
                pass
            else:
                self.unfix_dof(time_idx=t, m=m)
            

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


    def fix_dof_and_initialize(self, m=None):
        """
        Fix DOF for MP model and initialize steady-state models.
        """
        if m is None:
            m = self.m

        self.set_operating_conditions(m=m)
        self.initialize_system(m=m)


    def unfix_dof(self, time_idx=0, m=None):
        """
        Unfix linking variables in MP model
        """
        if m is None:
            m = self.m
        
        # Unfix feed flow rate
        m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
        # m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"].unfix()

        # Unfix P1 and P2 outlet pressures
        m.fs.P1.control_volume.properties_out[0].pressure.unfix()
        m.fs.P2.control_volume.properties_out[0].pressure.unfix()
        m.fs.M1.mixed_state[0].pressure.unfix()

        m.fs.dead_volume.dead_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
        m.fs.dead_volume.dead_volume.properties_out[0].flow_mass_phase_comp["Liq", "NaCl"].unfix()

        # m.fs.dead_volume.dead_volume.mass_frac_phase_comp[0, "Liq", "H2O"].unfix()
        # m.fs.dead_volume.dead_volume.mass_frac_phase_comp[0, "Liq", "NaCl"].unfix()

        # m.fs.dead_volume.dead_volume.properties_out[0].dens_mass_phase["Liq"].unfix()

        m.fs.dead_volume.dead_volume.properties_out[0].temperature.unfix()
        m.fs.dead_volume.dead_volume.properties_out[0].pressure.unfix()

        m.fs.dead_volume.delta_state.dens_mass_phase[0, "Liq"].unfix()
        m.fs.dead_volume.delta_state.mass_frac_phase_comp[0, "Liq", "NaCl"].unfix()

    
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

        # m.fs.P2.control_volume.properties_out[0].flow_vol_phase["Liq"].fix(
        #     pyunits.convert(self.recycle_flowrate, to_units=pyunits.m**3 / pyunits.s)
        # )

    #TODO
    # def build_flushing_system(self): -> Similar to RO system without a mixer
    # How is the flushing efficiency modeled in? 
    
    # def set_flushing_operating_conditions(self, m):
    # The feed flowrate for the flushing system should be fixed

    # Multiperiod constraints:
    # 1. Concentration at the end of the flushing period should be equal to the concentration at the start of the filtration period


if __name__ == "__main__":

    initial_conditions = {
        "n_time_points": 5,
        "raw_feed_conc": 5.8,
        "raw_feed_flowrate": 1.8,
        "recycle_flowrate": 49.911,
        "recycle_conc_start": 11.7,
        "temperature_start": 25,  # degC
        "p1_pressure_start": 306,  # psi
        "A_comp": 5.963600814843386e-12,
        "B_comp": 3.0790017613480806e-08,
        "channel_height": 0.0008636000000000001,
        "spacer_porosity": 0.85,
        # "cycle_pass_water_recovery": 0.063,
        "membrane_area": 7.9,
        "membrane_length": 1,
        "dead_volume": 0.01251,
        "accumulation_time": 60,
    }

    ccro = CCRO_dead_volume_flushing(**initial_conditions)

    ccro.create_multiperiod()
    try:
        ccro.solve(tee=False)
        print('Raw feed flowrate:', blks.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)
        print('Mixer outlet concentration:', blks.fs.M1.mixed_state[0].conc_mass_phase_comp["Liq", "NaCl"].value)
        print('Dead volume mass phase comp:', blks.fs.dead_volume.dead_volume.mass_phase_comp[0,"Liq", "NaCl"].value)
    

        
    except:
        # Print flowrate to RO model
        for blks in ccro.mp.get_active_process_blocks():
            print('Raw feed flowrate:', blks.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)
            print('Mixer outlet concentration:', blks.fs.M1.mixed_state[0].conc_mass_phase_comp["Liq", "NaCl"].value)
            print('Dead volume mass phase comp:', blks.fs.dead_volume.dead_volume.mass_phase_comp[0,"Liq", "NaCl"].value)
    

    # m = ccro.build_system()
    # m = ccro.set_operating_conditions(m)
    # ccro.initialize_system(m)

    # print("DOF =", degrees_of_freedom(m))
    # print("DOF FEED =", degrees_of_freedom(m.fs.raw_feed))
    # print("DOF PUMP 1 =", degrees_of_freedom(m.fs.P1))
    # print("DOF PUMP 2 =", degrees_of_freedom(m.fs.P2))
    # print("DOF DEAD VOLUME =", degrees_of_freedom(m.fs.dead_volume))
    # print("DOF MIXER =", degrees_of_freedom(m.fs.M1))
    # print("DOF RO =", degrees_of_freedom(m.fs.RO))

    # solver = get_solver()
    # results = solver.solve(m, tee=False)
    # assert_optimal_termination(results)

    # print('Calculated mass fraction:',ccro.recycle_flow_mass_salt()/(ccro.recycle_flow_mass_water() + ccro.recycle_flow_mass_salt()))

    # print('\nDelta State:')
    # m.fs.dead_volume.delta_state.mass_frac_phase_comp.display()

    # print("\nDead Volume:")
    # m.fs.dead_volume.dead_volume.mass_frac_phase_comp.display()

    # print("\nDead Volume Properties In:")
    # m.fs.dead_volume.dead_volume.properties_in[0].mass_frac_phase_comp.display()

    # print("\nDead Volume Properties Out:")
    # m.fs.dead_volume.dead_volume.properties_out[0].mass_frac_phase_comp.display()
    # m.fs.dead_volume.dead_volume.properties_out[0].flow_mass_phase_comp.display()

    # print('\nDOF in properties in:',degrees_of_freedom(m.fs.dead_volume.dead_volume.properties_in[0]))
    # print('\nDOF in properties out:',degrees_of_freedom(m.fs.dead_volume.dead_volume.properties_out[0]))
    # print('\nDOF in delta state:',degrees_of_freedom(m.fs.dead_volume.delta_state))
    # print('\nDOF in dead volume:',degrees_of_freedom(m.fs.dead_volume))

