import csv
import argparse
import math


def writeCSVFile(filename, data, dl):
    """Write data to csv file."""
    with open(filename, 'w', newline='') as output:
        csvWriter = csv.writer(output, delimiter=dl)
        for line in data:
            csvWriter.writerow(line)


def calcZ(distance):
    """Calculate the z data."""
    return math.sin(distance + 6) * 4 / pow(2.2, distance) + 3


def generateCheckerBoard(sampleCount, intervallSize, mid):
    """Create data with coordinates in a checkerboard style."""
    coordList = []
    samplesXAxis = -(-sampleCount // 2)
    samplesYAxis = samplesXAxis
    stepSize = ((2 * intervallSize) // sampleCount)
    distanceNorm = 12 / math.sqrt(math.pow(mid, 2) * 2)
    # global minium is located at 0. startX is an offset to move the first
    # value away from this point
    startX = 1.3
    for x in range(-samplesXAxis, samplesXAxis):
        for y in range(-samplesYAxis, samplesYAxis):
            # transform coordinates so, that mid is the point of origin
            xTransformed = x * (stepSize) + mid + stepSize / 2 + startX
            yTransformed = y * (stepSize) + mid + stepSize / 2 + startX
            # use distance from point of origin to calculate objective
            distance = math.dist([mid, mid], [xTransformed, yTransformed])
            distance *= distanceNorm
            z = calcZ(distance)
            # create a triplet of x, y, and z coordinates
            pair = (xTransformed, yTransformed, z)
            coordList.append(pair)
    return coordList


def generateCircular(sampleCount, intervallSize, mid, rootCount):
    """
    Create data with circular expanding coordinates around mid.

    Test data will be created in rings around the middle. To calculate the
    positions of data points on these rings, roots of unity are used for evenly
    distributed data points.
    """
    coordList = []
    radiusInc = intervallSize // (sampleCount // rootCount)
    phiIncrement = 2 * math.pi / rootCount
    # global minium is located at 0. startX is an offset to move the first
    # value away from this point
    startX = 1.3
    for radius in range(0, intervallSize, radiusInc):
        phi = 0
        r = radius + startX
        # calculate all roots of unity and their coordinates for the chosen
        # radius
        for x in range(0, rootCount):
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            # functions was approximated to 12, So each distance needs to be no
            distance = math.dist([0, 0], [x, y]) * 0.1875
            z = calcZ(distance)
            pair = (mid + x, mid + y, z)
            # add coordinates to the list and increment phi
            coordList.append(pair)
            phi = phi + phiIncrement
    return coordList


# create an argument parser
parser = argparse.ArgumentParser(
    description="Create test data for DSE benchmarks as csv file. The data " +
                "is generated in a way, that the global minimum will be in " +
                "the middle of the specified value space. The specified " +
                "value space will be used for the X and Y axes. The " +
                "resulting XY-plane will therefore be a square.")
# add arguments to the parser
parser.add_argument("output", type=ascii, help="Output file path. " +
                                               "Will be overwritten!")
parser.add_argument("delimiter", type=ascii,
                    help="Delimiter of data in the CSV. Default is ';'",
                    default=';', nargs='?')
parser.add_argument('start', type=int, metavar='START',
                    help='Start of value space', default=1, nargs=1)
parser.add_argument('end', type=int, metavar='END',
                    help='End of value space', default=1, nargs=1)
parser.add_argument('sampleCount', type=int, metavar='SAMPLECOUNT',
                    help='Count of data points to generate over the' +
                    ' whole intervall', default=1, nargs=1)
parser.add_argument('--circular', dest='circular', default=0, type=int,
                    metavar='ROOTCOUNT',
                    help='Create circular arranged data points. Each ring' +
                         ' has ROOTCOUNT data points.')
# parse arguments and make them available
args = parser.parse_args()
# remove leading and closing '' in string
output = args.output[1:-1]
sampleCount = args.sampleCount[0]
intervallSize = args.end[0] - args.start[0]
delimiter = args.delimiter[1:-1]
mid = 64
print(delimiter)

data = []
if 0 < args.circular:
    data = generateCircular(sampleCount, intervallSize, mid, args.circular)
else:
    data = generateCheckerBoard(sampleCount, intervallSize, mid)

writeCSVFile(output, data, delimiter)
print("Done: Wrote", len(data), "lines to", output)
