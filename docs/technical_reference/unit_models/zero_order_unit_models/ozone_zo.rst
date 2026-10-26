.. _ozone_zo:

Ozone  (ZO)
===========

Model Type
----------
This unit model is formulated as a **single-input, single-output** model form.
See documentation for :ref:`single-input, single-output Helper Methods<siso_methods>`.

Electricity Consumption
-----------------------
The constraint used to calculate energy consumption is described in the Additional Constraints section below. More details can be found in the unit model class.

Costing Method
--------------
Costing is calculated using the :py:meth:`~watertap.unit_models.zero_order.ozone_zo.OzoneZOData.cost_ozonation` method.
For full details on costing, see documentation for the :ref:`zero-order costing package<zero_order_costing>`.

        .. start_ozone_zo_costing

Ozone capital cost is calculated using the following equations.

.. math:: 
   C_{capital} = Cost\ Factor * (A + B * dose_{ozone} + C * ln(Q) + D * ln(Q) * dose_{ozone})

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the ozone contactor", ":math:`C_{capital}`", "USD"
   "Scaling factor to adjust the cost based on the size of the system", ":math:`Cost\ Factor`", "USD"
   "Coefficients determined from cost data", ":math:`A, B, C, D`", "Dimensionless"
   "Amount of ozone consumption", ":math:`dose_{ozone}`", "mg/L"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/h`"


        
Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Ozone contact time", "contact_time", ":math:`min`"
   "CT value for ozone contactor", "concentration_time", ":math:`mg*min/l`"
   "Ozone mass transfer efficiency", "mass_transfer_efficiency", ":math:`dimensionless`"
   "Specific energy consumption for ozone generation", "specific_energy_coeff", ":math:`kWh/lb`"
   "Mass flow rate of ozone", "ozone_flow_mass", ":math:`lb/h`"
   "Ozone consumption", "ozone_consumption", ":math:`mg/l`"
   "Ozone generation power demand", "electricity", ":math:`kW`"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Ozone consumption constraint", "ozone_consumption_constraint"
   "Ozone mass flow constraint", "ozone_flow_mass_constraint"
   "Ozone power constraint", "electricity_constraint"

.. index::
   pair: watertap.unit_models.zero_order.ozone_zo;ozone_zo

.. currentmodule:: watertap.unit_models.zero_order.ozone_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.ozone_zo
    :members:
    :noindex:
