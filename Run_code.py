from Image_Analysis import image_analysis, write_asymetry_to_file, write_maxima_to_file, write_maxima_to_file_2
import glob
from utils import parallel_process

#imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')

imgs = glob.glob('/disk1/ert/fits_images/*.fits')

out = parallel_process(imgs[0:20], image_analysis)
write_maxima_to_file_2('maxima_alt.txt', out)
write_maxima_to_file('maxima.txt', out)
write_asymetry_to_file('asymetry.txt', out)