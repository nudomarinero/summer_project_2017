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
from Image_Analysis import image_analysis

file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'

class MyTest(unittest.TestCase):
    def test_588013382727958540(self):
        # Test image: 588013382727958540.fits
        output = image_analysis(file_dir+'588013382727958540.fits')
        self.assertEqual(output[-1], False)

    def test_587739609175031857(self):
        # Test image: 587739609175031857.fits
        output = image_analysis(file_dir+'587739609175031857.fits')
        self.assertEqual(output[-1], False)

    def test_587727213348520295(self):
        # Test image: 587727213348520295.fits.fits
        output = image_analysis(file_dir+'587727213348520295.fits')
        self.assertEqual(output[-1], False)

    def test_587742061616758804(self):
        # Test image: 587742061616758804.fits.fits
        output = image_analysis(file_dir+'587742061616758804.fits')
        self.assertEqual(output[-1], True)

    def test_587736619321655535(self):
        # Test image: 587736619321655535.fits.fits
        output = image_analysis(file_dir+'587736619321655535.fits')
        self.assertEqual(output[-1], False)

    def test_587739167310807244(self):
        # Test image: 587742061616758804.fits.fits
        output = image_analysis(file_dir+'587739167310807244.fits')
        self.assertEqual(output[-1], False)

    def test_587733603734388952(self):
        # Test image:587733603734388952.fits.fits
        output = image_analysis(file_dir+'587733603734388952.fits')
        self.assertEqual(output[-1], True)

    def test_587739406805762069(self):
        # Test image: 587739406805762069.fits
        output = image_analysis(file_dir+'587739406805762069.fits')
        self.assertEqual(output[-1], False)

    def test_587734621629513866(self):
        # Test image: 587734621629513866.fits. Test case with only 1 maxima
        output = image_analysis(file_dir+'587734621629513866.fits')
        self.assertEqual(output[-1], False)

    def test_587744728761761895(self):
        # Test image: 587744728761761895.fits
        output = image_analysis(file_dir+'587744728761761895.fits')
        self.assertEqual(output[-1], False)

    def test_587742566784106610(self):
        # Test image: 587742566784106610.fits
        output = image_analysis(file_dir+'587742566784106610.fits')
        self.assertEqual(output[-1], False)

    def test_587730847428968484(self):
        # Test image: 587730847428968484.fits . Previously a false negativs
        output = image_analysis(file_dir+'587730847428968484.fits')
        self.assertEqual(output[-1], True)

    def test_587742572149080091(self):
        # Test image: 587730847428968484.fits . Previously a false negativs
        output = image_analysis(file_dir+'587730847428968484.fits')
        self.assertEqual(output[-1], True)

    def test_588017111293296655(self):
        # Test image: 588017111293296655.fits . 
        output = image_analysis(file_dir+'588017111293296655.fits')
        self.assertEqual(output[-1], True)

if __name__ == '__main__':
   unittest.main(warnings='ignore')