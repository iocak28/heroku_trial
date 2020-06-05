.. ECE 229 Project documentation master file, created by
   sphinx-quickstart on Mon Jun  1 16:26:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ECE 229 Group 6 Project's documentation!
===========================================


This project is a **Bank Telecaller Decision Support System**. 

The main purpose is to help bank manager and telecaller quickly target customers who are more likely to subscribe to their product.

Package Dependencies
--------------------

The project depends on plotly, dash, numpy, pandas, sklearn, xgboost and scikit-learn, etc. The packge version is listed below:

**For Data Analysis:**

.. code-block::

   ->plotly                        4.4.1 
   ->numpy                         1.16.4
   ->pandas                        0.24.2

**For Prediction:**

.. code-block::

   ->xgboost                       1.1.0
   ->sklearn                       0.0  
   ->scikit-learn                  0.22.2.post1
   ->pandas                        1.0.3
   ->numpy                         1.18.1

**For Dashboard Development:**

.. code-block::

   ->dash                          1.12.0             
   ->dash-core-components          1.10.0             
   ->dash-html-components          1.0.3              
   ->dash-renderer                 1.4.1              
   ->dash-table                    4.7.0
   ->plotly                        4.8.0

**For Test:**

.. code-block::

   ->coverage                      5.1 
   ->pytest                        5.0.1
   ->numpy                         1.16.4
   ->pandas                        0.24.2

.. toctree::
   :hidden:

   index

Data Analysis
==============    
.. automodule:: visualization.analysis
    :members:


Result Prediction
==============
.. automodule:: src.feature_extraction
    :members:
.. automodule:: src.pre_processing
    :members:
.. automodule:: src.prediction
    :members:

DashBoard 
==============
.. automodule:: dashboard
    :members:

.. automodule:: util
    :members:

Test
==============
.. automodule:: test.generate_coverage_report
    :members:

.. automodule:: test.test_analysis
    :members:

.. automodule:: test.test_dashboard
    :members:

.. automodule:: test.test_feature_extraction
    :members:

.. automodule:: test.test_pre_processing
    :members:

.. automodule:: test.test_prediction
    :members:

.. automodule:: test.test_util
    :members:






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
