summer_project_2017
===================

Detection and Classification of Galaxy Mergers
----------------------------------------------

The goal of this project was to develop an algorithm that could quickly
determine whether a galaxy was undergoing a merger or if it was isolated. The
asymmetry given by,

![Imgur](https://i.imgur.com/syGukgyt.png)

is used to classify the galaxies. Where I0 is the flux of an individual pixel of
the original image and Iθ is the flux of the same pixel location as the original
image after a 180 degree rotation around a chosen centroid. If A \> 0.2, the
galaxy is classified as a merger; non-merger otherwise.

Installation and documentation
------------------------------

Clone the project in the terminal by typing:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git git@github.com:nudomarinero/summer_project_2017.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the documentation:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd docs
make html
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage
-----

### Analysis of a single image - option 1

For analysis of a single image such as:

![Imgur](https://i.imgur.com/AbE3Eoyb.png)

-   First isolate the galaxy and remove any remaining small stars using

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
galaxy, galaxy_name = galaxy_isolation(image_dir, plot=False)
galaxy = remove_small_star(galaxy, plot=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This results in the image looking like

![Imgur](https://i.imgur.com/MBuQvkCb.png)

-   Then, for the above image, the location of maxima and the asymmetry are
    calculated using

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
maxima = find_local_maximum(galaxy, plot=False)
min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy, maxima, plot=False) 
detect_status = False
if len(maxima) == 1:
    detect_status = False
else:
    detect_status = detect_star(galaxy, plot=False) 
    if detect_status:
        galaxy_split = split_star_from_galaxy(galaxy, galaxy_name, plot=False)
        galaxy_split = remove_small_star(galaxy_split, plot=False)
        min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy_split, maxima, plot=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above gives the following data:

### Analysis of a single image - option 2

Alternatively, can just use

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from Image_Analysis import image_analysis

output_data = image_analysis(image_dir)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

where *image_dir* is the path to the image. This function returns the following
data:

1.  The name of the galaxy.

2.  The location and flux of the maxima.

3.  The value of the flux and shape asymmetry under a 180 and 90 degree rotation
    around the center.

4.  The value of the flux and shape asymmetry under a 180 degree rotation around
    the pixel that minimises the asymmetry.

5.  Whether or not a large star exists.

### Analysis of a large sample of images.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import glob
from tqdm import trange
from utils import parallel_process
from Image_Analysis import image_analysis

imgs = glob.glob('path_to_image_folder/*.fits')

step_size = 10000
nsteps = len(imgs)//step_size + 1
out = []
for k in trange(nsteps, desc="Blocks"):
 low_limit = k*step_size
 high_limit = (k+1)*step_size
 out += parallel_process(imgs[low_limit:high_limit], image_analysis, 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This results in a list of lists with each index corresponding to the output from
`image_analysis()`.

 
