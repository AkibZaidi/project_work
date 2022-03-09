from excelParser import csv2Dataframe, df2MsExcel, df2Dict, meshGrid2Interpolate
import scriptRun
import sys, os


def main():
    if sys.argv[1] == "run":
        dataFrame = csv2Dataframe('results.csv')
        df2MsExcel(dataFrame,'trafficPatternType',['Bitcomplement', 'Bitrevers', 'Bitrotate','Bitshuffle','Transpose1', 'Transpose2'])
        dfDict = df2Dict('results.xlsx','Bitcomplement')
        meshGrid2Interpolate(dfDict)
    elif sys.argv[1] == "clean":
        os.system("rm -rf *.xlsx")
        os.system("rm -rf *.swp")
    else:
        print("Provide Valid Input")


if __name__ == '__main__':
    main()