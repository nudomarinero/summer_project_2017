summer_project_2017
===================

 

Detection and Classification of Galaxy Mergers
----------------------------------------------

 

The goal of this project was to develop an algorithm that could quickly
determine whether a galaxy was undergoing a merger or if it was isolated. The
asymmetry given by,

\\begin{figure}  
\\includegraphics[width=300pt, height = 125 pt]{latex-image-1.png}  
\\end{figure}

is used to classify the galaxies. If `A > 0.2`, the galaxy is classified as a
merger; non-merger otherwise.

 

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

 

For analysis of a single image such as:

![](Report/Before_smoothing.png)

-   First isolate the galaxy and remove any remaining small stars using

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
galaxy, galaxy_name = galaxy_isolation(image_dir, plot=False)
galaxy = remove_small_star(galaxy, plot=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This results in the image looking like

-   Then, for the above image, the location of maxima and the asymmetry are
    calculated using

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
maxima = find_local_maximum(galaxy, plot=False)
min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy, maxima, plot=False) 
detect_status = False
if len(maxima) == 1:
 detect_status = False
else:
 detect_status = detect_star(galaxy, plot=False) # 
 if detect_status:
 galaxy_split = split_star_from_galaxy(galaxy, galaxy_name, plot=False)
 galaxy_split = remove_small_star(galaxy_split, plot=False)
 min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy_split, maxima, plot=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, can just use

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from Image_Analysis import image_analysis

output_data = image_analysis(image_dir)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

where *image_dir* is the path to the image.

 

 
