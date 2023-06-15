===========
box_jenkins
===========


.. image:: https://img.shields.io/pypi/v/boxjenkins.svg
        :target: https://pypi.python.org/pypi/boxjenkins

.. image:: https://img.shields.io/travis/dbedollac/boxjenkins.svg
        :target: https://github.com/dbedollac/boxjenkins

.. image:: https://readthedocs.org/projects/boxjenkins/badge/?version=latest
        :target: https://boxjenkins.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Introduction
------------


box_jenkins contains all the tools you need to define an ARIMA model for a time series based on the methodology proposed by Box and Jenkins, moreover it includes a function to find an ARIMA model following this methodology automatically. The selected model will fulfill the required statistical assumptions:

    * Mean of residuals statistically equal to zero.
    * Residuals described by white noise process.
    * Roots of the lag polynomials out of the unit circle.
    * Lag polynomials without approximately common factors.
    * Principle of parsimony.

In comparison with most of other auto_arima solutions, this approach tends to find models with fewer parameters (avoiding overfitting) and a comparable or better accuracy.

Installation
------------

::

   pip install box_jenkins
