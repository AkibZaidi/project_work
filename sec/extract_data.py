#!/usr/bin/env python3.7

from parasweep import run_sweep, CartesianSweep
from parasweep.dispatchers import SubprocessDispatcher
from tqdm import tqdm
import os
import numpy as np
import csv

def extractValue(element, sign):
    return element.partition(sign)[2].replace(" ", "")

# open the file in the write mode
csvFile = open('results.csv', 'w', encoding='UTF8')
# create the csv writer
writer = csv.writer(csvFile)

header = ['bufferSize', 'packetLength', 'packetInjectionRate','avgLat','minLat','maxLat']
writer.writerow(header)
config = "config"
results = "results"
lst_config = os.listdir(config)
lst_config.sort()
lst_results = os.listdir(results)
lst_results.sort()

data = []

for filename in lst_config:
    mylines = []                                # Declare an empty list.
    with open(os.path.join(config, filename), 'rt') as myfile:    # Open lorem.txt for reading text.
        for myline in myfile:                   # For each line in the file,
            # strip newline and add to list.
            mylines.append(myline.rstrip('\n'))
    for element in mylines:                     # For each element in the list,
        if element and element[0] != "#":
            if not element.find("bufferSize"):
                bufferSize = int(extractValue(element, "="))
            if not element.find("packetLength"):
                packetLength = int(extractValue(element, "="))
            if not element.find("packetInjectionRate"):
                packetInjectionRate = float(extractValue(element, "="))
    data.append([bufferSize,packetLength,packetInjectionRate])
    
counter = 0
for filename in lst_results:
    mylines = []                                # Declare an empty list.
    with open(os.path.join(results, filename), 'rt') as myfile:    # Open lorem.txt for reading text.
        for myline in myfile:                   # For each line in the file,
            # strip newline and add to list.
            mylines.append(myline.rstrip('\n'))
    for element in mylines:                     # For each element in the list,
        if element and element[0] != "#":
            if not element.find("Average latency (cycles)"):
                avgLat = int(extractValue(element, ":"))
            if not element.find("Minimum latency (cycles)"):
                minLat = int(extractValue(element, ":"))
            if not element.find("Maximum latency (cycles)"):
                maxLat = int(extractValue(element, ":"))
    data[counter].append(avgLat);
    data[counter].append(minLat);
    data[counter].append(maxLat);
    counter +=1
    # write a row to the csv file
    #writer.writerow([bufferSize,packetLength,packetInjectionRate])
# write data to the csv file
writer.writerows(data)  
# close the file
csvFile.close()
