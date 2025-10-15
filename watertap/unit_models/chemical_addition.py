#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
Chemical addition unit model
"""

from pyomo.environ import (
    Var,
    Param,
    units as pyunits,
)
from pyomo.common.config import ConfigValue, In

from idaes.core import (
    declare_process_block_class,
)
from idaes.core.util.misc import StrEnum
from idaes.core.util.constants import Constants
import idaes.logger as idaeslog
from idaes.models.unit_models.statejunction import StateJunctionData

from watertap.costing.unit_models.chemical_addition import cost_chemical_addition

__author__ = "Kurban Sitterley"

solution_dens_dict = {
    "ammonia": 900,
    "alum": 1360,
    "anti_scalant": 1021,
    "caustic": 1540,
    "ferric_chloride": 1460,
    "hydrochloric_acid": 1490,
    "sodium_hypochlorite": 1300,
    "lime": 1250,
    "polymer": 1100,
    "soda_ash": 2200,
    "sodium_bisulfite": 1480,
    "sulfuric_acid": 1781,
}

ratio_in_solution_dict = {
    "ammonia": 0.3,
    "alum": 0.5,
    "anti_scalant": 1,
    "caustic": 0.5,
    "ferric_chloride": 0.42,
    "hydrochloric_acid": 0.37,
    "sodium_hypochlorite": 1,
    "lime": 0.5,
    "polymer": 0.1,
    "soda_ash": 1,
    "sodium_bisulfite": 1,
    "sulfuric_acid": 1,
}


class ChemicalType(StrEnum):
    ammonia = "ammonia"
    alum = "alum"
    anti_scalant = "anti_scalant"
    caustic = "caustic"
    ferric_chloride = "ferric_chloride"
    hydrochloric_acid = "hydrochloric_acid"
    hydrazine = anti_scalant
    sodium_hypochlorite = "sodium_hypochlorite"
    hypochlorite = sodium_hypochlorite
    lime = "lime"
    polymer = "polymer"
    sodium_bisulfite = "sodium_bisulfite"
    sodium_hydroxide = "sodium_hydroxide"
    soda_ash = "soda_ash"
    sulfuric_acid = "sulfuric_acid"


@declare_process_block_class("ChemAddition")
class ChemAdditionData(StateJunctionData):
    """
    Chemical addition unit model
    """

    CONFIG = StateJunctionData.CONFIG()

    CONFIG.declare(
        "chemical",
        ConfigValue(
            domain=In(ChemicalType),
            default=None,
            description="Type of chemical addition",
            doc="""Indicates the type of chemical addition to be modeled,
    **default** = None.""",
        ),
    )

    def build(self):

        super().build()

        self.solution_density = Param(
            initialize=solution_dens_dict[self.config.chemical],
            mutable=True,
            units=pyunits.g / pyunits.liter,
            doc=f"Density of {self.config.chemical} solution",
        )

        self.ratio_in_solution = Param(
            initialize=ratio_in_solution_dict[self.config.chemical],
            mutable=True,
            units=pyunits.dimensionless,
            doc=f"Mass fraction of {self.config.chemical} in solution",
        )

        self.pump_head = Param(
            initialize=10,
            units=pyunits.meter,
            mutable=True,
            doc=f"Pump head for {self.config.chemical} addition",
        )

        self.pump_efficiency = Param(
            initialize=0.75,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Pump efficiency",
        )

        self.dose = Var(
            initialize=1,
            units=pyunits.g / pyunits.liter,
            bounds=(0, None),
            doc=f"Dose of {self.config.chemical} addition",
        )

        self.chemical_flow_vol = Var(
            initialize=1,
            units=pyunits.m**3 / pyunits.s,
            bounds=(0, None),
            doc=f"Volumetric flow rate of {self.config.chemical} solution",
        )

        self.pumping_power = Var(
            initialize=1,
            units=pyunits.kW,
            bounds=(0, None),
            doc=f"Pumping power for {self.config.chemical} addition",
        )

        @self.Expression(
            doc=f"Mass flow rate of {self.config.chemical} added",
        )
        def chemical_mass_flow(b):
            return pyunits.convert(
                b.chemical_flow_vol
                * b.solution_density
                * b.ratio_in_solution,
                to_units=pyunits.kg / pyunits.s,
            )

        @self.Constraint(
            doc=f"Chemical dosage constraint for {self.config.chemical} addition",
        )
        def eq_chemical_flow_vol(b):
            return b.chemical_flow_vol == pyunits.convert(
                b.dose
                * b.properties[0].flow_vol_phase["Liq"]
                / (b.ratio_in_solution * b.solution_density),
                to_units=pyunits.m**3 / pyunits.s,
            )

        @self.Constraint(
            doc=f"Pumping power constraint for {self.config.chemical} addition",
        )
        def eq_pumping_power(b):
            return b.pumping_power == pyunits.convert(
                b.chemical_mass_flow
                * b.pump_head
                * Constants.acceleration_gravity
                / b.pump_efficiency,
                to_units=pyunits.kW,
            )

    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):

        super().initialize_build(state_args, outlvl, solver, optarg)

    @property
    def default_costing_method(self):
        return cost_chemical_addition
