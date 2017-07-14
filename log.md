Log
===

Log of changes, to-dos and goals

2017-07-14
----------

### Sky background level and threshold
Use the sigma clipping method of astropy.
The threshold can be deffined as mean+std.

If we find problems in the future we could try a Gaussian fit or a manual 
iterative approach.

Next steps:
* Separate galaxies. Try Watershed-like algorithm with erosion.
* Look for facility where to run the algorithms in parallel.
* Use binary masks instead of pixel intensity.

2017-07-11
----------
Galaxy selection algorithm completed.

See how to select the threshold level.

TODO:
* Use Python 3

2017-07-10
----------
Reproduce method described in: 
[https://ui.adsabs.harvard.edu/#abs/2016MNRAS.456.3032P/abstract]

General plan:
* Setup software
* Data:
 * Define test sample
* Algorithms:
 * Development
 * Validation (Galaxy Zoo, etc)
* Report