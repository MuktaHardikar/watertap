ZO Costing Functions
====================

Costing functions for zero-order unit models that do not follow power law..

anaerobic_mbr_mec_zo
--------------------
.. start_anaerobic_mbr_mec_zo_costing

Anaerobic MBR MEC capital cost and operating cost is calculated using the following equation:

.. math:: 
   C_{capital} = Cost\ Factor * (Unit\_Capex * Q)

.. math:: 
   C_{fixed\_opex} = Unit\_Opex * Q

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the anaerobic MBR MEC", "C_{capital}", ":math:`USD`"
   "Fixed operating cost of the anaerobic MBR MEC", "C_{fixed\_opex}", ":math:`USD`"
   "Scaling factor to adjust the capital cost based on the size of the system", "Cost\ Factor", ":math:`USD`"
   "Capital cost per unit flow of the anaerobic MBR MEC", "Unit\_Capex", ":math:`USD/(L/day)`"
   "Operating cost per unit flow of the anaerobic MBR MEC", "Unit\_Opex", ":math:`USD/(L/day)`"
   "Volumetric flow rate of the influent water", "Q", ":math:`m^{3}/h`"

.. end_anaerobic_mbr_mec_zo_costing


autothermal_hydrothermal_liquefaction
-------------------------------------
.. start_autothermal_hydrothermal_liquefaction_costing


.. end_autothermal_hydrothermal_liquefaction_costing


ozone_zo
--------
.. start_ozone_zo_costing

Ozone capital cost is calculated using the following equations.

.. math:: 
   C_{capital} = Cost\ Factor * (A + B * dose_{ozone} + C * ln(Q) + D * ln(Q) * dose_{ozone})

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the ozone contactor", ":math:`C_{capital}`",  ":math:`USD`"
   "Scaling factor to adjust the cost based on the size of the system", ":math:`Cost\ Factor`", ":math:`USD`"
   "Coefficients determined from cost data", ":math:`A, B, C, D`", ":math:`Dimensionless`"
   "Amount of ozone consumption", ":math:`dose_{ozone}`", ":math:`mg/L`"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/h`"

.. end_ozone_zo_costing


peracetic_acid_disinfection_zo
------------------------------
.. start_peracetic_acid_disinfection_zo



.. end_peracetic_acid_disinfection_zo


photothermal_membrane_zo
------------------------
.. start_photothermal_membrane_zo



.. end_photothermal_membrane_zo

pump_electricity_zo
-------------------
.. start_pump_electricity_zo



.. end_pump_electricity_zo


sedimentation_zo
----------------
.. start_sedimentation_zo



.. end_sedimentation_zo


storage_tank_zo
---------------
.. start_storage_tank_zo



.. end_storage_tank_zo


struvite_classifier_zo
----------------------
.. start_struvite_classifier_zo



.. end_struvite_classifier_zo

suboxic_activated_sludge_zo
---------------------------
.. start_suboxic_activated_sludge_zo



.. end_suboxic_activated_sludge_zo


supercritical_salt_precipitation_zo
-----------------------------------
.. start_supercritical_salt_precipitation_zo

Capital cost is calculated using the following equations:

.. math:: 
   C_{capital} = Cost\ Factor * A * B * (Q/C})^D

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the supercritical salt precipitation unit", ":math:`C_{capital}`", ":math:`USD`"
   "Installation factor", ":math:`A`", ":math:`Dimensionless`"
   "Equipment cost", ":math:`B`", ":math:`USD`"
   "Volumetric flow rate of the influent water", ":math:`Q`",  ":math:`lb/hr`"
   "Base volumetric flowrate", ":math:`C`", ":math:`lb/hr`"
   "Scaling exponent", ":math:`D`", ":math:`Dimensionless`"

.. end_supercritical_salt_precipitation_zo


surface_discharge_zo
--------------------
.. start_surface_discharge_zo_costing

Surface discharge capital cost is calculated using the following equations:

.. math:: 
   C_{capital} = Cost\ Factor * (C_{capital,near\_shore\_discharge} + C_{capital,pipe})

.. math:: 
   C_{near\_shore\_discharge} = A * (Q/Q_{ref)^B

.. math:: 
   C_{pipe\_cost} = Pipe\ Cost\ Basis * pipe\_distance * pipe\_diameter

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the UV-AOP unit", ":math:`C_{capital}`", ":math:`USD`"
   "Near shore discharge cost component of the UV-AOP unit capital cost", ":math:`C_{capital,near\_shore\_discharge}`", ":math:`USD`"
   "Pipe component of the well field capital", ":math:`C_{capital,pipe}`", ":math:`USD`"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/day`"
   "Reference volumetric flow rate of the influent water", ":math:`Q_{ref}`", ":math:`m^{3}/day`"
   "Coefficients determined from cost data", ":math:`A, B`", ":math:`USD, Dimensionless`"

..  end_surface_discharge_zo_costing

uv_aop_zo
---------
.. start_uv_aop_zo_costing

UV-AOP unit capital cost is calculated using the following equations:

.. math:: 
   C_{capital} = Cost\ Factor * ((A * Q + B * Electricity) + (C * chemical\_flow^{D} ))

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the UV-AOP unit", ":math:`C_{capital}`", ":math:`USD`"
   "Reactor cost determined from cost data", ":math:`A`", ":math:`USD/m^{3}/h`"
   "Lamp cost determined from cost data", ":math:`B`", ":math:`USD/kW`"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/h`"
   "Electricity ", "Electricity", "kW"
   "AOP capital cost factors", ":math:`C , D`", ":math:`USD, Dimensionless`"
   "Chemical flow", ":math:`chemical\_flow`", ":math:`lbs/day`"

.. end_uv_aop_zo_costing


uv_zo
-----
.. start_uv_zo_costing

UV capital cost is calculated using the following equations:

.. math:: 
   C_{capital} = Cost\ Factor * (A * Q + B * Electricity)

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the UV system", ":math:`C_{capital}`", ":math:`USD`"
   "Reactor cost determined from cost data", ":math:`A`", ":math:`USD/m^{3}/h`"
   "Lamp cost determined from cost data", ":math:`B`", ":math:`USD/kW`"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/h`"
   "Electricity ", "", "kW"

.. end_uv_zo_costing


vfa_recovery_zo
---------------
.. start_vfa_recovery_zo_costing

Capital cost is calculated using the following equation:

.. math:: 
   C_{capital} = Cost\ Factor * (Unit\_Capex * Q)

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the VFA recovery unit", ":math:`C_{capital}`", ":math:`USD`"
   "Capital cost per unit flow of the VFA recovery unit", "Unit\_Capex", ":math:`USD/(L/day)`"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`L/day`"

.. end_vfa_recovery_zo_costing


well_field_zo
-------------
.. start_well_field_zo_costing

Well field capital cost is calculated using the following equations:

.. math:: 
   C_{capital} = Cost\ Factor * (C_{capital,well} + C_{capital,pipe})

.. math::
   C_{capital,pipe} = Pipe\ Cost\ Basis * pipe\_distance * pipe\_diameter

.. math::
   C_{capital,well} = A * Q^{B}

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the well field", ":math:`C_{capital}`", ":math:`USD`"
   "Scaling factor to adjust the capital cost based on the size of the system", ":math:`Cost\ Factor`", ":math:`USD`"
   "Well component of the well field capital", ":math:`C_{capital,well}`", ":math:`USD`"
   "Pipe component of the well field capital", ":math:`C_{capital,pipe}`", ":math:`USD`"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/h`"
   "Cost per unit length and unit diameter of the pipe", ":math:`Pipe\ Cost\ Basis`", ":math:`USD/(mile*inch)`"
   "Coefficients determined from cost data", ":math:`A, B`", ":math:`USD, Dimensionless`"

.. end_well_field_zo_costing
