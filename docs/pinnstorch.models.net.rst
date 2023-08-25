Neural Nets
===========

We have two different neural nets. `FCN` and `NetHFM`.

.. automodule:: pinnstorch.models.net.neural_net
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Here's how to use the class provided by this module:

.. code-block:: python

   from pinnstorch.models.net.neural_net import FCN

   net = FCN(
    layers = [2, 50, 50, 2],
    lb = mesh.lb,
    ub = mesh.ub,
    output_names = ['u', 'v'],
    discrete = False
   )
