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

import pyomo.environ as pyo
from ..util import (
    register_costing_parameter_block,
    make_capital_cost_var,
    make_fixed_operating_cost_var,
)


def build_alum_cost_param_block(blk):
    # CatCost v 1.1.1
    # Aluminum sulphate, 5-lb. bgs., c.l., works, frt. equald., 17% Al203, W. Coast
    blk.cost = pyo.Var(
        initialize=0.54,
        doc="Alum cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Alum purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=15408,
        doc="Alum addition capital cost A parameter",
        units=pyo.units.USD_2007,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.5479,
        doc="Alum addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("alum", blk.cost / blk.purity)


def build_ammonia_cost_param_block(blk):
    # CatCost v 1.1.1
    # Ammonia, US Gulf, spot c.f.r. Tampa
    blk.cost = pyo.Var(
        initialize=0.76,
        doc="Ammonia cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Ammonia purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=6699.1,
        doc="Ammonia addition capital cost A parameter",
        units=pyo.units.USD_2020,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.4219,
        doc="Ammonia addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("ammonia", blk.cost / blk.purity)


def build_caustic_cost_param_block(blk):
    # CatCost v 1.1.1
    # Caustic soda (sodium hydroxide), liq., dst contract f.o.b.
    blk.cost = pyo.Var(
        initialize=0.92,
        doc="Caustic soda cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=0.5,  # assumed
        doc="Caustic soda purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=2262.8,
        doc="Caustic soda addition capital cost A parameter",
        units=pyo.units.USD_2007,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.6195,
        doc="Caustic soda addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("caustic_soda", blk.cost / blk.purity)


def build_ferric_chloride_cost_param_block(blk):
    # CatCost v 1.1.1
    # Ferric chloride, technical grade, 100% basis, tanks, f.o.b. works
    blk.cost = pyo.Var(
        initialize=1053.68,
        doc="Ferric chloride cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Ferric chloride purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=34153,
        doc="Ferric chloride addition capital cost A parameter",
        units=pyo.units.USD_2020,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.319,
        doc="Ferric chloride addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("ferric_chloride", blk.cost / blk.purity)


def build_hydrochloric_acid_cost_param_block(blk):
    # CatCost v 1.1.1
    # Hydrochloric acid, 22 deg. Be, US Gulf dom. ex-works US NE
    blk.cost = pyo.Var(
        initialize=0.12,
        doc="HCl cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=0.36,  # 22 deg. Be
        doc="HCl purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=900.97,
        doc="Hydrochloric acid addition capital cost A parameter",
        units=pyo.units.USD_2020,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.6179,
        doc="Hydrochloric acid addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("hydrochloric_acid", blk.cost / blk.purity)


def build_lime_cost_param_block(blk):
    # CatCost v 1.1.1
    # Lime, hydrated, bulk, t.l., f.o.b. works
    blk.cost = pyo.Var(
        initialize=0.11,
        doc="Lime cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Lime purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=12985,
        doc="Lime addition capital cost A parameter",
        units=pyo.units.USD_2020,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.5901,
        doc="Lime addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("lime", blk.cost / blk.purity)


# def build_polymer_cost_param_block(blk):

#     blk.cost = pyo.Var(
#         initialize=0.17,
#         doc="Polymer cost",
#         units=pyo.units.USD_2020 / pyo.units.kg,
#     )
#     blk.purity = pyo.Var(
#         initialize=1,
#         doc="Polymer purity",
#         units=pyo.units.dimensionless,
#     )
#     blk.capital_A_parameter = pyo.Var(
#         initialize=15408,
#         doc="Polymer addition capital cost A parameter",
#         units=pyo.units.USD_2007,
#     )
#     blk.capital_b_parameter = pyo.Var(
#         initialize=0.5479,
#         doc="Polymer addition capital cost b parameter",
#         units=pyo.units.dimensionless,
#     )

#     costing = blk.parent_block()
#     costing.register_flow_type("polymer", blk.cost / blk.purity)


def build_soda_ash_cost_param_block(blk):
    # CatCost v 1.1.1
    # Soda ash (sodium carbonate), dense, US Gulf, f.o.b. bulk
    blk.cost = pyo.Var(
        initialize=0.23,
        doc="Soda ash cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Soda ash purity",
        units=pyo.units.dimensionless,
    )
    # TODO: Check these values; adopted from ferric chloride
    blk.capital_A_parameter = pyo.Var(
        initialize=34153,
        doc="Soda ash addition capital cost A parameter",
        units=pyo.units.USD_2020,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.319,
        doc="Soda ash addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("soda_ash", blk.cost / blk.purity)


def build_sodium_bisulfite_cost_param_block(blk):

    blk.cost = pyo.Var(
        initialize=0.17,
        doc="Sodium bisulfite cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Sodium bisulfite purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=900.97,
        doc="Sodium bisulfite addition capital cost A parameter",
        units=pyo.units.USD_2007,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.6179,
        doc="Sodium bisulfite addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("sodium_bisulfite", blk.cost / blk.purity)


def build_sodium_hypochlorite_cost_param_block(blk):

    blk.cost = pyo.Var(
        initialize=0.17,
        doc="Sodium hypochlorite cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Sodium hypochlorite purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=900.97,
        doc="Hypochlorite addition capital cost A parameter",
        units=pyo.units.USD_2007,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.6179,
        doc="Hypochlorite addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("sodium_hypochlorite", blk.cost / blk.purity)


def build_sulfuric_acid_cost_param_block(blk):

    blk.cost = pyo.Var(
        initialize=0.17,
        doc="Sulfuric acid cost",
        units=pyo.units.USD_2020 / pyo.units.kg,
    )
    blk.purity = pyo.Var(
        initialize=1,
        doc="Sulfuric acid purity",
        units=pyo.units.dimensionless,
    )
    blk.capital_A_parameter = pyo.Var(
        initialize=900.97,
        doc="Sulfuric acid addition capital cost A parameter",
        units=pyo.units.USD_2007,
    )
    blk.capital_b_parameter = pyo.Var(
        initialize=0.6179,
        doc="Sulfuric acid addition capital cost b parameter",
        units=pyo.units.dimensionless,
    )

    costing = blk.parent_block()
    costing.register_flow_type("sulfuric_acid", blk.cost / blk.purity)


def cost_chemical_addition(blk):

    chem_build_rule_dict = {
        "ammonia": build_ammonia_cost_param_block,
        "lime": build_lime_cost_param_block,
        "ferric_chloride": build_ferric_chloride_cost_param_block,
        "soda_ash": build_soda_ash_cost_param_block,
        "alum": build_alum_cost_param_block,
        # "polymer": build_polymer_cost_param_block,
        "caustic": build_caustic_cost_param_block,
        "sulfuric_acid": build_sulfuric_acid_cost_param_block,
        "sodium_hypochlorite": build_sodium_hypochlorite_cost_param_block,
        "sodium_bisulfite": build_sodium_bisulfite_cost_param_block,
        "hydrochloric_acid": build_hydrochloric_acid_cost_param_block,
    }

    chemical = blk.unit_model.config.chemical
    chem_build_rule = chem_build_rule_dict.get(chemical, None)
    if chem_build_rule is None:
        raise ValueError(f"Unrecognized chemical type {chemical} in ChemAddition")

    @register_costing_parameter_block(
        build_rule=chem_build_rule, parameter_block_name=chemical
    )
    def cost_chem_addition(blk):
        make_capital_cost_var(blk)
        blk.costing_package.add_cost_factor(blk, "TPEC")
        make_fixed_operating_cost_var(blk)

    cost_chem_addition(blk)
