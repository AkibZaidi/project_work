from parasweep import run_sweep, CartesianSweep
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows
import csv
import sys, os
from collections import OrderedDict
def main():
    bufferSize_list = [30,32,36,30,32,36,36,32,48,96,56,64,64,72,76,72,80,88,92,96,112,88,116,120]
    #bufferSize_list = [30,32]
    packetLength_List = [8,16,32,64,128]
    trafficPattern_list = ['Bitcomplement']
    count =0
    run_sim = []
    packet_run = []
    buffer_run = []
    trafficPattern_run = []
    replaceLatencyDict = OrderedDict()
    if sys.argv[1] == "run":
        for trafficPattern in trafficPattern_list:
            for bufferSize in bufferSize_list:
                for packetLength in packetLength_List:
                    mapping = run_sweep(
                        command='./panaca -c ./{sim_id}.txt',
                        configs=['{sim_id}.txt'],
                        templates=['config_asir.txt'],
                        sweep=CartesianSweep({  'var_packetLength': [packetLength],
                                        'var_bufferSize': [bufferSize],
                                        'var_trafficPattern' : [trafficPattern]
                                     }),
                        verbose=True,
                        sweep_id='panaca_results')
                    df = pd.read_csv('results.csv', header = 0, sep=";")
                    df1 = df[['bufferSize', 'packetLength', 'Minimum latency', 'Maximum latency', 'Average latency']]
                    #df1 = df
                    #print(df1)
            df1['avgLatency'] = df1['Average latency'].str.extract('(\d+)').astype(float)
            #f1['Average latency'].replace(df1['Average latency'].str.extract('(\d+)').astype(float))
            print(df1['avgLatency'])
            #print(df1)
            df1['timePerFlit'] = df1['avgLatency']/(df1['packetLength'])
            df1.to_excel('result_xl.xlsx',sheet_name=trafficPattern, index=False)

    elif sys.argv[1] == "clean":
        os.system("rm -rf results.csv")
        os.system("rm -rf p*.txt")
    else:
        print("Please provide an input")
        print(sys.argv)

if __name__ == '__main__':
    main()
