import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt

import json
from collections import OrderedDict
from scipy.interpolate import griddata
from scipy.io import savemat

def main():
  result_xl = extract_data('result_xl.xlsx', 'Sheet1')
  dictToJson(result_xl)
  meshGrid2Interpolate(result_xl)

def extract_data(path, sheet):
  '''
  read_data will take the path of the excel/csv/text data
  file and return an Ordered Dictionary with the data wrapped 
  inside it
  '''
  dfToDict = OrderedDict()
  dataFrame = pd.read_excel(path, sheet_name = sheet,header=0, dtype={
    "Average latency (ns)": float,
    "Packet Length": float,
    "Buffer Size": float,
    "Time per Flit": float

  })
  bufferSizeList   = dataFrame['Buffer Size'].to_list()
  packetLengthList = dataFrame['Packet Length'].to_list()
  timePerFlitList      = dataFrame['Time per Flit'].to_list()

  dfToDict['bufferSize'] = bufferSizeList
  dfToDict['packetLength'] = packetLengthList
  dfToDict['timePerFlit'] = timePerFlitList
  print(dataFrame)
  return dfToDict
  
def dictToJson(dictToConvert):
  '''
  converts Dictionary to Json and writes a JSON data file
  return : No return, creates a json file
  '''
  json_object = json.dumps(dictToConvert, indent = 4)
  with open("data.json", "w") as outfile:
    outfile.write(json_object)

def meshGrid2Interpolate(jsonData):
  xq,yq = np.mgrid[0:1:600, 0:1:128]
 
  bufferSize = np.array(jsonData['bufferSize']).T
  packetLength = np.array(jsonData['packetLength']).T
  timePerFlit = np.array(jsonData['timePerFlit']).T

  xy = np.zeros((2,np.size(bufferSize)))
  xy[0] = bufferSize
  xy[1] = packetLength
  xy = xy.T
  x_array = np.arange(0,601)
  y_array = np.arange(0,128)

  #grid_x, grid_y = np.meshgrid(x_array, y_array)
  #grid_x, grid_y = np.mgrid[0.0:0.09:601*1j, 0.0:0.03:128*1j]
  grid_x, grid_y = np.mgrid[0:601:1, 0:128:1]
  # print(grid_x)
  # print(grid_y)
  print('XY')
  print(xy)
  print('Timeperflit')
  print(timePerFlit)
  print('Grid Z')
  grid_z = griddata(xy, timePerFlit, (grid_x, grid_y), method='nearest')

  dse_array = OrderedDict()
  dse_array['bufferSize'] = bufferSize
  dse_array['packetLength'] = packetLength
  dse_array['timePerFlit'] = timePerFlit
  dse_array['vq'] = grid_z
  savemat("dse_array.mat", dse_array)

  print(grid_z)
  min_timeperflit = np.argmin(np.argmin(grid_z))
  print(min_timeperflit)
  # plot
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # plt.contourf(bufferSize,timePerFlit,grid_z,np.arange(0,601,1))
  # plt.plot(bufferSize,timePerFlit,'k.')
  # plt.xlabel('xi',fontsize=16)
  # plt.ylabel('yi',fontsize=16)
  # plt.savefig('interpolated.png',dpi=100)
  # plt.close(fig)


  return tuple(ans)
if __name__ == '__main__':
    main()
