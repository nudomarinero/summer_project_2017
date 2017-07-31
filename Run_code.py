import glob
import os
from Image_Analysis import image_analysis, write_asymetry_to_file, write_maxima_to_file, write_maxima_to_file_2
from utils import parallel_process

imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')

# imgs = glob.glob('/disk1/ert/fits_images/*.fits')

out = parallel_process(imgs[0:30], image_analysis)
write_maxima_to_file_2('auto_test_maxima_alt2.txt', out)
write_maxima_to_file('auto_test_maxima2.txt', out)
write_asymetry_to_file('auto_test_asymetry2.txt', out)

os.system('git add auto*.txt')
os.system('git commit -m "Output auto upload"')
os.system('git push')