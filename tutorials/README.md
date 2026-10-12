# WaterTAP Tutorials & Examples

This directory contains tutorial Jupyter notebooks for WaterTAP.
# WaterTAP Tutorials

## Basic

- [WaterTAP Introduction Tutorial](./introduction.ipynb)
  > A quick introduction to Pyomo, followed by examples that illustrate the application of WaterTAP property models, unit models, flowsheet models, and cost optimization.

- [Using a Simple Property Model](./using_a_simple_property_model.ipynb)
  > A quick introduction to using a simple property model in WaterTAP. Properties are modeled separately from process unit models, lending to the modularity and flexibility of modeling in WaterTAP.

- [Using a Simple Unit Model](./using_a_simple_unit_model.ipynb)
  > A quick introduction to using a simple unit model in WaterTAP. Unit models are modeled separately from property models, lending to the modularity and flexibility of modeling in WaterTAP.

- [Introduction to the Reverse Osmosis Unit Model](./ro/reverse_osmosis_0D.ipynb)
  > An introduction to using WaterTAP's reverse osmosis model with its simplest configuration settings. Usage of the isothermal NaCl property model is also demonstrated.

- [Introduction to the Low-salt-rejection Reverse Osmosis (LSRRO) Flowsheet Model](./lsrro.ipynb)
  > An introduction to using WaterTAP's LSRRO model for high-salinity desalination.

## Intermediate

- [Creating a Simple Property Model](./creating_a_simple_property_model.ipynb)
  > A quick introduction to creating a simple property model in WaterTAP.

- [Creating a Simple Unit Model](./creating_a_simple_unit_model.ipynb)
  > A quick introduction to creating a simple unit model in WaterTAP.

- [Fitting Membrane Parameters for Reverse Osmosis](./RO_Param_Fitting/RO_param_fitting.ipynb)
  > A tutorial on how to use WaterTAP to determine membrane properties from experimental data and spec sheets.

## Advanced

- [Benchmark Simulation Model No. 2 (BSM2)](./BSM2/BSM2.ipynb)
  > A demonstration on how to build, initialize, and simulate the flowsheet for Benchmark Simulation Model No. 2 (BSM2), a process model for representing a full wastewater treatment plant.

- [Conducting a Parameter Estimation](./parmest_demo/parmest_example.ipynb)
  > Using Pyomo's `parmest` tool, this tutorial shows how to conduct parameter estimation to obtain an empirical equation for the vapor pressure of an NaCl solution over ranges of temperature and salt mass fraction values.

- [Modeling Multicomponent Rejection with Nanofiltration](./multicomponent_rejection_nf/modeling_multicomponent_rejection.ipynb)
  > A tutorial on how to build, initialize, and simulate a flowsheet that utilizes the Donnan Steric Pore Model w/Dielectric Exclusion (DSPM-DE) for predicting multicomponent rejection with nanofiltration. Usage of the multicomponent aqueous solution (MCAS) property model is also demonstrated.
  
- [OLI Calculations in WaterTAP](./oli_calculations.ipynb)
  > A tutorial that shows how to login and make calls to the OLI Cloud API for advanced aqueous chemistry calculations. **Note:** an OLI license is required to fully run this tutorial and  any OLI API functionality within WaterTAP.

- [Parameter Sweep Tool: Getting Started](./parameter_sweep_demo.ipynb)
  > A tutorial on using WaterTAP's "parameter sweep," an analysis tool for performing optimization, sensitivity analyses, and uncertainty quantification.

- [Unit Model Customization](./unit_model_customization_example.ipynb)
  > A tutorial on customizing unit models at the flowsheet level.

- [External Model Integration](./model_integration/model_integration_workflow.ipynb)
  > A tutorial on how to integrate external models with WaterTAP using QSDsan (Quantitative Sustainable Design for Sanitation & Resource Recovery Systems) as an example.

- [Flexible Desalination](./flex_ro/mp_model.ipynb)  
  > A tutorial on how to optimize the cost of a Pricetaker model given a dataset of price signals and the system's operating conditions.