.. _well_field_zo:

Well Field (ZO)
===============

Model Type
----------
This unit model is formulated as a **pass-through** model form.
See documentation for :ref:`pass-through Helper Methods<pt_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the **pump_electricity** helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the :py:meth:`~watertap.unit_models.zero_order.well_field_zo.WellFieldZOData.cost_well_field` method.
For full details on costing, see documentation for the :ref:`zero-order costing package<zero_order_costing>`.

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

   "Capital cost of the well field", ":math:`C_{capital}`", "USD"
   "Scaling factor to adjust the capital cost based on the size of the system", ":math:`Cost\ Factor`", "USD"
   "Capital cost of the well component of the well field", ":math:`C_{capital,well}`", "USD"
   "Capital cost of the pipe component of the well field", ":math:`C_{capital,pipe}`", "USD"
   "Volumetric flow rate of the influent water", ":math:`Q`", ":math:`m^{3}/h`"
   "Cost per unit length and unit diameter of the pipe", ":math:`Pipe\ Cost\ Basis`", "USD/(mile*inch)"
   "Coefficients determined from cost data", ":math:`A, B`", "USD, Dimensionless"


        
Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Piping distance", "pipe_distance", ":math:`mi`"
   "Pipe diameter", "pipe_diameter", ":math:`in`"

.. index::
   pair: watertap.unit_models.zero_order.well_field_zo;well_field_zo

.. currentmodule:: watertap.unit_models.zero_order.well_field_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.well_field_zo
    :members:
    :noindex:
