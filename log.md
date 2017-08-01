Log
===

Log of changes, to-dos and goals

2017-08-01
----------
The new algorithm that minimises the asymmetry looking around the central peak
of the image is working OK. We find some clearly wrong high values for images 
that contain text (they come from failed SQL queries). The galaxies with high
values of the asymmetry (above about 0.8) are typicaly produced by intruding 
bringht stars in the image.

Next steps:
* Create set of test images with high asymmetries with stars and genuine.
* Use additional data and data from the detected peaks to separate the wrong
 assymetries.
* Consider testing the goodness of fit to a Gaussian to the possible stars.
* Presentation.

2017-07-24
----------
Preliminary version of the peak detector ready. Code divided in methods. First
draft of the tests.

Next steps:
* Complete the docstrings of the methods.
* Look at [sphinx](http://www.sphinx-doc.org) for the documentation.
* Write tests.
* Save flux values at peaks when running on the full sample.

2017-07-18
----------
Binnary mask A_{180} classifier finished and speed improved.

Next steps:
* Run on 2000 galaxy sample. Output csv with values.
  * Optinal but useful: pixel Asymmetry values and A_{90}
* Plot relation between A_{180} and P_MG

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