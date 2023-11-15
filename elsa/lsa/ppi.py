import compcore
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import scipy as sp
import h5py
import sys, csv, re, os, time, argparse, string, tempfile
try:
  import lsalib
except ImportError:
  from lsa import lsalib

def main():
    parser = argparse.ArgumentParser()
    arg_precision_default=1000
    arg_delayLimit_default=0

    parser.add_argument("dataFile", metavar="dataFile", type=argparse.FileType('r'), \
        help="the input data file,\n \
        m by (r * s)tab delimited text; top left cell start with \
        '#' to mark this is the header line; \n \
        m is number of variables, r is number of replicates, \
        s it number of time spots; \n \
        first row: #header  s1r1 s1r2 s2r1 s2r2; \
        second row: x  ?.?? ?.?? ?.?? ?.??; for a 1 by (2*2) data")
    parser.add_argument("resultFile", metavar="resultFile", type=argparse.FileType('w'), \
        help="the output result file")
    parser.add_argument("-e", "--extraFile", dest="extraFile", default=None, \
        type=argparse.FileType('r'),
        help="specify an extra datafile, otherwise the first datafile will be used \n \
                and only lower triangle entries of pairwise matrix will be computed")
    parser.add_argument("-d", "--delayLimit", dest="delayLimit", default=arg_delayLimit_default, type=int,\
        help="specify the maximum delay possible, default: {},\n \
                must be an integer >=0 and <spotNum".format(arg_delayLimit_default))
    parser.add_argument("-m", "--minOccur", dest="minOccur", default=50, type=int, 
        help="specify the minimum occurence percentile of all times, default: 50,\n")
    parser.add_argument("-r", "--repNum", dest="repNum", default=1, type=int,
        help="specify the number of replicates each time spot, default: 1,\n \
                must be provided and valid. ")
    parser.add_argument("-s", "--spotNum", dest="spotNum", default=4, type=int, 
        help="specify the number of time spots, default: 4,\n \
                must be provided and valid. ")
    parser.add_argument("-p", "--pvalueMethod", dest="pvalueMethod", default="perm", \
        choices=["perm", "theo", "mix"],
        help="specify the method for p-value estimation, \n \
                default: pvalueMethod=perm, i.e. use  permutation \n \
                theo: theoretical approximaton; if used also set -a value. \n \
                mix: use theoretical approximation for pre-screening \
                if promising (<0.05) then use permutation. ")
    parser.add_argument("-x", "--precision", dest="precision", default=arg_precision_default, type=int,\
        help="permutation/precision, specify the permutation \n \
                number or precision=1/permutation for p-value estimation. \n \
                default is {}, must be an integer >0 ".format(arg_precision_default) )
    parser.add_argument("-b", "--bootNum", dest="bootNum", default=0, type=int, \
        choices=[0, 100, 200, 500, 1000, 2000],
        help="specify the number of bootstraps for 95%% confidence \
                interval estimation, default: 100,\n \
                choices: 0, 100, 200, 500, 1000, 2000. \n \
                Setting bootNum=0 avoids bootstrap. \n \
                Bootstrap is not suitable for non-replicated data.")
    parser.add_argument("-t", "--transFunc", dest="transFunc", default='simple', \
        choices=['simple', 'SD', 'Med', 'MAD'],\
        help="specify the method to summarize replicates data, default: simple, \n \
                choices: simple, SD, Med, MAD                                     \n \
                NOTE:                                                             \n \
                simple: simple averaging                                          \n \
                SD: standard deviation weighted averaging                         \n \
                Med: simple Median                                                \n \
                MAD: median absolute deviation weighted median;" )
    parser.add_argument("-f", "--fillMethod", dest="fillMethod", default='none', \
        choices=['none', 'zero', 'linear', 'quadratic', 'cubic', 'slinear', 'nearest'], \
        help="specify the method to fill missing, default: none,               \n \
                choices: none, zero, linear, quadratic, cubic, slinear, nearest  \n \
                operation AFTER normalization:  \n \
                none: fill up with zeros ;   \n \
                operation BEFORE normalization:  \n \
                zero: fill up with zero order splines;           \n \
                linear: fill up with linear splines;             \n \
                slinear: fill up with slinear;                   \n \
                quadratic: fill up with quadratic spline;             \n \
                cubic: fill up with cubic spline;                \n \
                nearest: fill up with nearest neighbor") 
    parser.add_argument("-n", "--normMethod", dest="normMethod", default='robustZ', \
        choices=['percentile', 'percentileZ', 'pnz', 'robustZ', 'rnz', 'none'], \
        help="must specify the method to normalize data, default: robustZ, \n \
                choices: percentile, none, pnz, percentileZ, robustZ or a float  \n \
                NOTE:                                                   \n \
                percentile: percentile normalization, including zeros (only with perm)\n \
                pnz: percentile normalization, excluding zeros (only with perm) \n  \
                percentileZ: percentile normalization + Z-normalization \n \
                rnz: percentileZ normalization + excluding zeros + robust estimates (theo, mix, perm OK) \n \
                robustZ: percentileZ normalization + robust estimates \n \
                (with perm, mix and theo, and must use this for theo and mix, default) \n")
    parser.add_argument("-q", "--qvalueMethod", dest="qvalueMethod", \
        default='scipy', choices=['scipy'],
        help="specify the qvalue calculation method, \n \
                scipy: use scipy and storeyQvalue function, default \n \
                ")
                #R: use R's qvalue package, require X connection")
    parser.add_argument("-T", "--trendThresh", dest="trendThresh", default=None, \
        type=float, \
        help="if trend series based analysis is desired, use this option \n \
                NOTE: when this is used, must also supply reasonble \n \
                values for -p, -a, -n options")
    parser.add_argument("-a", "--approxVar", dest="approxVar", default=1, type=float,\
        help="if use -p theo and -T, must set this value appropriately, \n \
                precalculated -a {1.25, 0.93, 0.56,0.13 } for i.i.d. standard normal null \n \
                and -T {0, 0.5, 1, 2} respectively. For other distribution \n \
                and -T values, see FAQ and Xia et al. 2013 in reference")
    parser.add_argument("-v", "--progressive", dest="progressive", default=0, type=int, 
        help="specify the number of progressive output to save memory, default: 0,\n \
                2G memory is required for 1M pairwise comparison. ")
    arg_namespace = parser.parse_args()

    fillMethod = vars(arg_namespace)['fillMethod']
    normMethod = vars(arg_namespace)['normMethod']
    qvalueMethod = vars(arg_namespace)['qvalueMethod']
    pvalueMethod = vars(arg_namespace)['pvalueMethod']
    precision = vars(arg_namespace)['precision']
    transFunc = vars(arg_namespace)['transFunc']
    bootNum = vars(arg_namespace)['bootNum']
    approxVar = vars(arg_namespace)['approxVar']
    trendThresh = vars(arg_namespace)['trendThresh']
    progressive = vars(arg_namespace)['progressive']
    delayLimit = vars(arg_namespace)['delayLimit']
    minOccur = vars(arg_namespace)['minOccur']
    dataFile = vars(arg_namespace)['dataFile']				#dataFile
    extraFile = vars(arg_namespace)['extraFile']				#extraFile
    resultFile = vars(arg_namespace)['resultFile']			#resultFile
    repNum = vars(arg_namespace)['repNum']
    spotNum = vars(arg_namespace)['spotNum']

    try:
        extraFile_name = extraFile.name 
    except AttributeError:
        extraFile_name = ''

    assert trendThresh==None or trendThresh>=0

    if transFunc == 'SD':
        fTransform = lsalib.sdAverage
    elif transFunc == 'Med':
        fTransform = lsalib.simpleMedian
    elif transFunc == 'MAD':
        fTransform = lsalib.madMedian   
    else:
        fTransform = lsalib.simpleAverage 

    if repNum < 5 and transFunc == 'SD':
        print("Not enough replicates for SD-weighted averaging, fall back to simpleAverage", file=sys.stderr)
        transFunc = 'simple'
    if repNum < 5 and transFunc == 'MAD':
        print("Not enough replicates for Median Absolute Deviation, fall back to simpleMedian", file=sys.stderr)
        transFunc = 'Med'

    if normMethod == 'none':
        zNormalize = lsalib.noneNormalize
    elif normMethod == 'percentile':
        zNormalize = lsalib.percentileNormalize
    elif normMethod == 'percentileZ':
        zNormalize = lsalib.percentileZNormalize
    elif normMethod == 'robustZ':
        zNormalize = lsalib.robustZNormalize
    elif normMethod == 'pnz':
        zNormalize = lsalib.noZeroNormalize
    elif normMethod == 'rnz':
        zNormalize = lsalib.robustNoZeroNormalize
    else:
        zNormalize = lsalib.percentileZNormalize

    start_time = time.time()

    col = spotNum
    total_row_0 = 0
    total_row_1 = 0
    block = 2000
    first_file = "first_file.txt"
    second_file = "second_file.txt"

    with open(first_file, 'r') as textfile:
        next(textfile)
        for line in textfile:
            total_row_0 += 1
    with open(second_file, 'r') as textfile:
        next(textfile)
        for line in textfile:
            total_row_1 += 1

    i_m = 0
    j_m = 0
    start_0 = 1
    end_0 = block
    start_1 = 1
    end_1 = block

    if end_0 >= total_row_0:
        end_0 = total_row_0
    if end_1 >= total_row_1:
        end_1 = total_row_1
        
    manager = multiprocessing.Manager()
    first_Data = manager.list()
    second_Data = manager.list()

    while i_m * block < total_row_0:
        i_m += 1
        skip_header = start_0
        skip_footer = total_row_0 - end_0
        firstData = np.genfromtxt(first_file, comments='#', delimiter='\t',missing_values=['na', '', 'NA'], filling_values=np.nan,usecols=range(1,spotNum*repNum+1), skip_header=skip_header, skip_footer=skip_footer)
        if len(firstData.shape) == 1:
            data = np.array([firstData])

        firstFactorLabels = np.genfromtxt(first_file, comments='#', delimiter='\t', usecols=range(0,1), dtype='str', skip_header=skip_header, skip_footer=skip_footer).tolist()
        if type(firstFactorLabels)==str:
            firstFactorLabels=[firstFactorLabels]

        factorNum = firstData.shape[0]
        tempData=np.zeros( ( factorNum, repNum, spotNum), dtype='float' ) 
        for i in range(0, factorNum):
            for j in range(0, repNum):
                try:
                    tempData[i,j] = firstData[i][np.arange(j,spotNum*repNum,repNum)]
                except IndexError:
                    print("Error: one input file need more than two data row or use -e to specify another input file", file=sys.stderr)
                    quit()

        for i in range(0, factorNum):
            for j in range(0, repNum):
                tempData[i,j] = lsalib.fillMissing( tempData[i,j], fillMethod )
        first_Data.append(tempData)
    while j_m * block < total_row_1:
        j_m += 1
        skip_header = start_1
        skip_footer = total_row_1 - end_1
        secondData = np.genfromtxt(second_file, comments='#', delimiter='\t',missing_values=['na', '', 'NA'], filling_values=np.nan,usecols=range(1,spotNum*repNum+1), skip_header=skip_header, skip_footer=skip_footer)
        if len(secondData.shape) == 1:
            data = np.array([secondData])

        secondFactorLabels=np.genfromtxt( second_file, comments='#', delimiter='\t', usecols=range(0,1), dtype='str', skip_header=skip_header, skip_footer=skip_footer).tolist()
        if type(secondFactorLabels)==str:
            secondFactorLabels=[secondFactorLabels]

        factorNum = secondData.shape[0]
        tempData=np.zeros((factorNum,repNum,spotNum),dtype='float')
        for i in range(0, factorNum):
            for j in range(0, repNum):
                try:
                    tempData[i,j] = secondData[i][np.arange(j,spotNum*repNum,repNum)]
                except IndexError:
                    print("Error: one input file need more than two data row or use -e to specify another input file", file=sys.stderr)
                    quit()
        for i in range(0, factorNum):
            for j in range(0, repNum):
                tempData[i,j] = lsalib.fillMissing( tempData[i,j], fillMethod )
        second_Data.append(tempData)

    merged_filename = 'merged_data_1.h5'
    def myfun_pall(i):
        data = compcore.LSA(total_row_0, total_row_1)
        for j in range(0, len(second_Data)):
            array = lsalib.palla_applyAnalysis( first_Data[i], second_Data[j], data, col, onDiag=True, delayLimit=delayLimit,bootNum=bootNum, pvalueMethod=pvalueMethod, 
                                                precisionP=precision, fTransform=fTransform, zNormalize=zNormalize, approxVar=approxVar, resultFile=resultFile, trendThresh=trendThresh, 
                                                firstFactorLabels=firstFactorLabels, secondFactorLabels=secondFactorLabels, qvalueMethod=qvalueMethod, progressive=progressive)
            with h5py.File(merged_filename, 'w') as merged_hf:
                merged_hf.create_dataset(f'data_{i}_{j}', data=array)
        return 1
    

    pool = multiprocessing.Pool(processes=10)

    results = [pool.apply_async(myfun_pall, args=(process_id,)) for process_id in range(len(second_Data))]

    for result in results:
        a = result.get()


    # parallel_obj = Parallel(n_jobs= -1)
    # parallel_obj(delayed(myfun_pall)(i) for i in range(0, len(first_Data)))

    print("finishing up...", file=sys.stderr)
    end_time=time.time()
    print("time elapsed %f seconds" % (end_time - start_time), file=sys.stderr)

if __name__=="__main__":
    main()
