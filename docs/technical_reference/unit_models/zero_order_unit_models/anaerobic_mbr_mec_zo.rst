.. _anaerobic_mbr_mec_zo:

Integrated Anaerobic Membrane Bioreactor/Microbial Electrolysis Cell (ZO)
=========================================================================

Model Type
----------
This unit model is formulated as a **reactive single-inlet, double-outlet** model form.
See documentation for :ref:`reactive single-inlet, double-outlet Helper Methods<sidor_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the **pump_electricity** helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the :py:meth:`~watertap.unit_models.zero_order.anaerobic_mbr_mec_zo.AnaerobicMBRMECZOData.cost_anaerobic_mbr_mec` method.
For full details on costing, see documentation for the :ref:`zero-order costing package<zero_order_costing>`.

        .. start_anaerobic_mbr_mec_zo_costing

Anaerobic MBR MEC capital cost and operating cost is calculated using the following equation:

.. math:: 
   C_{capital} = Cost\ Factor * (Unit\_Capex * Q)

.. math:: 
   C_{fixed\_opex} = Unit\_Opex * Q

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Capital cost of the anaerobic MBR MEC", "C_{capital}", "USD"
   "Fixed operating cost of the anaerobic MBR MEC", "C_{fixed\_opex}", "USD"
   "Scaling factor to adjust the capital cost based on the size of the system", "Cost\ Factor", "USD"
   "Capital cost per unit flow of the anaerobic MBR MEC", "Unit\_Capex", "USD/(L/day)"
   "Operating cost per unit flow of the anaerobic MBR MEC", "Unit\_Opex", "USD/(L/day)"
   "Volumetric flow rate of the influent water", "Q", ":math:`m^{3}/h`"


        
.. index::
   pair: watertap.unit_models.zero_order.anaerobic_mbr_mec_zo;anaerobic_mbr_mec_zo

.. currentmodule:: watertap.unit_models.zero_order.anaerobic_mbr_mec_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.anaerobic_mbr_mec_zo
    :members:
    :noindex:
