'''
Created on 21 Jul 2017

@author: Sahl
'''
# import sys
import sys
import unittest
import warnings
# from scipy import ndimage as ndi
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import morphology
from skimage import measure
import numpy.ma as ma
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import glob
from astropy.io import fits

sys.path.append('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/')
# print(sys.path)
from Image_Analysis import image_analysis, detect_star

file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'

class MyTest(unittest.TestCase):
    def test_is_star_in_588013382727958540(self):
        """
        Checks if there is a star in image 588013382727958540.fits. This image
        has 2 objects that are clearly galaxies, and 1 small and very circular
        object with no clear diffraction spikes. Thus even if the small object
        is a star, it is not relevant.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'588013382727958540.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587739609175031857(self):
        """
        Checks if there is a star in image 587739609175031857.fits. This image
        has 2 objects that are clearly galaxies; where 1 galaxy is small while
        the other is large. This image has no stars.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'587739609175031857.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)    
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587727213348520295(self):
        """
        Checks if there is a star in image 587727213348520295.fits. This image
        has a large spiral galaxy with a bright center and very small local
        maxima. This image has no stars.

        This test will fail if a star is detected.
        """       
        output = image_analysis(file_dir+'587727213348520295.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587742061616758804(self):
        """
        Checks if there is a star in image 587742061616758804.fits. This image
        has a large spiral galaxy with a faint center next to a very bright
        star with diffraction spikes.

        This test will fail if a star is not detected.
        """
        output = image_analysis(file_dir+'587742061616758804.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, True, msg='Star not detected when it should have been.')

    def test_is_star_in_587736619321655535(self):
        """
        Checks if there is a star in image 587736619321655535.fits. This image
        has multiple large objects with small bright centers and large halos.
        This image has no stars.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'587736619321655535.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587739167310807244(self):
        """
        Checks if there is a star in image 587739167310807244.fits. This image
        has 2 large objects that are clearly galaxies. This image has no stars.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'587739167310807244.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587733603734388952(self):
        """
        Checks if there is a star in image 587733603734388952.fits. This image
        has a large object with multiple nuclei at the center. There is also a 
        large bright star with clear diffraction spikes next to it.

        This test will fail if a star is not detected.
        """
        output = image_analysis(file_dir+'587733603734388952.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, True, msg='Star not detected when it should have been.')

    def test_is_star_in_587739406805762069(self):
        """
        Checks if there is a star in image 587739406805762069.fits. This image
        has 2 large objects that are clearly galaxies. This image has no stars.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'587739406805762069.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587734621629513866(self):
        """
        Checks if there is a star in image 587734621629513866.fits. This image
        has only a galaxy with a single maxima. This image has no stars.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'587734621629513866.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587744728761761895(self):
        """
        Checks if there is a star in image 587744728761761895.fits. This image
        has a large irregular object at the center. This image has no stars.

        This test will fail if a star is detected.
        """    
        output = image_analysis(file_dir+'587744728761761895.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587742566784106610(self):
        """
        Checks if there is a star in image 587742566784106610.fits. This image
        has 2 small objects that are galaxies. This image has no stars.

        This test will fail if a star is detected.
        """
        output = image_analysis(file_dir+'587742566784106610.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, False, msg='Star detected when it should not have been.')

    def test_is_star_in_587730847428968484(self):
        """
        Checks if there is a star in image 587730847428968484.fits. This image
        has a large elliptical galaxy with a faint center next to a very bright
        star with diffraction spikes.

        This test will fail if a star is not detected.
        """
        output = image_analysis(file_dir+'587730847428968484.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, True, msg='Star not detected when it should have been.')

    def test_is_star_in_587742572149080091(self):
        """
        Checks if there is a star in image 587742572149080091.fits. This image
        has 3 small circular objects. One of them is a bright star with clear
        diffrection spikes. One is a very small circular object. And at the
        center is a slightly faint galaxy the same size as the star.

        This test will fail if a star is not detected.
        """
        output = image_analysis(file_dir+'587742572149080091.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, True, msg='Star not detected when it should have been.')

    def test_is_star_in_588017111293296655(self):
        """
        Checks if there is a star in image 588017111293296655.fits. This image
        has a small circular galaxy at the center. It also has a very large star
        with large diffraction spikes.

        This test will fail if a star is not detected.
        """
        output = image_analysis(file_dir+'588017111293296655.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, True, msg='Star not detected when it should have been.')

    def test_is_star_in_588007006334943294(self):
        """
        Checks if there is a star in image 588007006334943294.fits. This image
        has a galaxy with a small bright center, but a huge halo. This image
        also has a small bright star with diffraction spikes within tht edges of
        the galaxy halo.

        This test will fail if a star is not detected.
        """
        output = image_analysis(file_dir+'588007006334943294.fits')
        min_asmmetry_flux, maxima, galaxy_name, galaxy = output[5], output[1], output[0], output[-1]
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy)  
        self.assertEqual(detect_status, True, msg='Star not detected when it should have been.')

if __name__ == '__main__':
    unittest.main(warnings='ignore')