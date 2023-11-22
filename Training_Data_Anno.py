# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 12:16:28 2023

@author: Nathan Gilmore
"""

import numpy as np
import random
import glob
from astropy.io import fits
from meteostat import Point, Daily, Hourly
from datetime import datetime
import os
import shutil
import time

weather = np.array(['Clear', 'Fair', 'Cloudy', 'Overcast', 'Fog', 'Freezing Fog', 'Light Rain', 'Rain', 'Heavy Rain', 'Freezing Rain', 'Heavy Freezing Rain', 'Sleet', 'Heavy Sleet', 'Light Snowfall', 'Snowfall', 'Heavy Snowfall', 'Rain Shower', 'Heavy Rain Shower', 'Sleet Shower', 'Heavy Sleet Shower', 'Snow Shower', 'Heavy Snow Shower', 'Lightning', 'Hail', 'Thunderstorm', 'Heavy Thunderstorm', 'Storm'])
no_weather_data = 0
out_of_time_range = 0
images_processed = 0
ambiguous_images = 0
clear_images = 0
obscured_images = 0
meteo_weather = 'Null'
coco = 0
fits_dir = r'/beegfs/car/njm/cloud_class/data/fits/'
jpg_dir = r'/beegfs/car/dcampbell/allsky/camera1/jpeg'

def count_files(dir_path):
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

#lists all items in the directory
#checks if  they are FILES or DIRECTORIES
#returns the number of FILES

def find_file(filename, root_dir):
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)

#searches through all files, in all subdirectories, within the parent directory
#checks if the given filename is among the files it has found
#returns the full filepath of that file

def get_weather_data (year, month, day, hour):    
        global coco
        global meteo_weather
        #Set time period of image ready for meteostat
        start = datetime(year, month, day, hour)
        end = datetime(year, month, day, hour)

        #Create Point for Bayfordbury, UK foe meteostat 
        bayfordbury = Point(51.775, -0.094444)

        #Obtain Image Weather info from Meteostat
        data = Hourly(bayfordbury, start, end) #obtains the data from meteostat for a given date and time
        data = data.fetch() #puts this data into a pandas DataFrame
        clcv=data.iat[0,10] #takes the 0th row, 10th column value (in this case, weather code)
        i = int(clcv)-1
        coco = i+1
        meteo_weather=weather[i] #returns a weather condition as indicated by the weather code

def sort_images (random_file):
        global clear_images
        global obscured_images
        global ambiguous_images
        global coco
        #print(coco)
        #print(meteo_weather)
    #Split filename and extension
        file = random_file.split('.',1)[0] #takes the filename without the ".FIT" extension
        #print(file)
    #Copy .jpg version of clear fits images    
        if int(coco) == 1:
            clear_images = clear_images + 1
            #put image into clear folder
            new_dir = r'/beegfs/car/njm/cloud_class/clear_images'
            file_path = find_file(file+'.JPG', jpg_dir) #looks for the image in the jpg directory
            shutil.copy2(file_path,new_dir) #copies the file from "file_path", into "new_dir"
    #Ignore fits images decleared as 'Fair'
        if int(coco) == 2:
            ambiguous_images=ambiguous_images + 1
    #Copy .jpg version of obscured images, ignoring those defined as 'fair'
        if int(coco) == 3:
            obscured_images = obscured_images + 1
            #put image into cloudy folder
            new_dir = r'/beegfs/car/njm/cloud_class/cloudy_images'
            file_path = find_file(file+'.JPG', jpg_dir) #looks for the image in the jpg directory
            shutil.copy2(file_path,new_dir) #copies the file from "file_path", into "new_dir"
        else:
            ambiguous_images=ambiguous_images + 1
        return clear_images, obscured_images, ambiguous_images

#Loop as necessary
file_list = os.listdir(fits_dir)    #creates a list of all of the files in the directory
for unused in range(0, count_files(fits_dir)):     #will loop all of the images in the directory
    dt0 = time.time()
    images_processed = images_processed + 1
    #Get a random FIT Image from FIT Directory, 
    random_file = random.choice(file_list) 
    #print (random_file)
    dt1 = time.time()
    #print('Time to chose fits:',dt1 - dt0)

    #Extract timestamp from fits header
    with fits.open(fits_dir+'/'+random_file, mode="update") as hdul:
        hdul.info()
        hdul[0].verify('fix') #ensures the HDU data is is consistent and correct and will attempt to fix if not
        hdul[0].header
        datestamp = hdul[0].header['DATE-OBS'] #takes the datestamp of the FITS image
        #print (datestamp)
        #splitting the datestamp into its year, month, day, hour, and minute components
        year = int(datestamp[0:4])
        month = int(datestamp[5:7])
        day = int(datestamp[8:10])
        hour = int(datestamp[11:13])
        minutes = int(datestamp[14:16])

        dt2 = time.time()
        #print('Time to open fits:',dt2 - dt1)
  
    #Only process images 10mins either side of the hour
    if minutes <= 10 or minutes >= 50:  #good solution to this, i tidied it up to make it a smidgen faster
        if minutes >= 50:
            hour=hour+1
        if hour < 24:  # e.g. if time 23:55:00, it would be rounded to 24:55:00 which is invalid
            try:
                get_weather_data(year,month,day,hour)
                dt3 = time.time()
                #print('Time to get weather data:',dt3 - dt2)

                hdul[0].header.append('WEATHER', meteo_weather)
                sort_images(random_file)

                dt4 = time.time()
                #print('Time to move things',dt4 - dt3)
            except:
                no_weather_data = no_weather_data + 1
    else:
        out_of_time_range = out_of_time_range + 1
        
#Print Summary
#print ('Number of Images Processed ', images_processed)
#print ('Outside of requisite time-frames ', out_of_time_range)
#print ('Ambiguous images ', ambiguous_images)
#print ('Clear Images ', clear_images)
#print ('Obscured Images ', obscured_images)
#print ('No Weather Data ', no_weather_data)
