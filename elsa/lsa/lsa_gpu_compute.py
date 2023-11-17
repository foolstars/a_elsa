#lsa-compute -- computation script for LSA package to perform lsa table calculation 

#License: BSD

#Copyright (c) 2008 Li Charles Xia
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#1. Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#3. The name of the author may not be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
#IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
#NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
#THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from tqdm import tqdm
import compcore
import numpy as np
import scipy as sp
import h5py
import sys, csv, re, os, time, argparse, string, tempfile
try:
  import lsalib
except ImportError:
  from lsa import lsalib

rpy_import = False

def main():
    # __script__ = "lsa_compute"
    # version_desc = os.popen("lsa_version").read().rstrip()
    # version_print = "%s (rev: %s) - copyright Li Charlie Xia, lixia@stanford.edu" % (__script__, version_desc) 
    # print(version_print, file=sys.stderr)

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
    block = 3000

    next(dataFile)
    for line in dataFile:
        total_row_0 += 1

    next(extraFile)
    for line in extraFile:
        total_row_1 += 1

    if qvalueMethod in ['R'] and rpy_import:
        qvalue_func = lsalib.R_Qvalue
    else:
        qvalue_func = lsalib.storeyQvalue 

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

    data = compcore.LSA(total_row_0, total_row_1)
    outer_total = total_row_0 // block
    outer_desc = "total_task"
    inner_total = total_row_1 // block
    inner_desc = "inner_task"

    pall_array = []
    merged_filename = 'merged_data_1.h5'
    with h5py.File(merged_filename, 'w') as merged_hf:
        with tqdm(total=outer_total, desc=outer_desc) as outer_bar:
            while i_m * block < total_row_0:
                skip_header = start_0
                skip_footer = total_row_0 - end_0
                firstData = np.genfromtxt(dataFile.name, comments='#', delimiter='\t',missing_values=['na', '', 'NA'], filling_values=np.nan,usecols=range(1,spotNum*repNum+1), skip_header=skip_header, skip_footer=skip_footer)

                if len(firstData.shape) == 1:
                    firstData = np.array([firstData])

                firstFactorLabels = np.genfromtxt(dataFile.name, comments='#', delimiter='\t', usecols=range(0,1), dtype='str', skip_header=skip_header, skip_footer=skip_footer).tolist()
                if type(firstFactorLabels)==str:
                    firstFactorLabels=[firstFactorLabels]

                cleanData = []
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
                cleanData.append(tempData)

                with tqdm(total=inner_total, desc=inner_desc, leave=False) as inner_bar:
                    while j_m * block < total_row_1:
                        
                        skip_header = start_1
                        skip_footer = total_row_1 - end_1
                        secondData = np.genfromtxt(extraFile.name, comments='#', delimiter='\t',missing_values=['na', '', 'NA'], filling_values=np.nan,usecols=range(1,spotNum*repNum+1), skip_header=skip_header, skip_footer=skip_footer)

                        if len(secondData.shape) == 1:
                            secondData = np.array([secondData])

                        secondFactorLabels=np.genfromtxt(extraFile.name, comments='#', delimiter='\t', usecols=range(0,1), dtype='str', skip_header=skip_header, skip_footer=skip_footer).tolist()
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
                        cleanData.append(tempData)

                        array = lsalib.palla_applyAnalysis( cleanData[0], cleanData[1], data, col, onDiag=True, delayLimit=delayLimit,bootNum=bootNum, pvalueMethod=pvalueMethod, 
                                                            precisionP=precision, fTransform=fTransform, zNormalize=zNormalize, approxVar=approxVar, resultFile=resultFile, trendThresh=trendThresh, 
                                                            firstFactorLabels=firstFactorLabels, secondFactorLabels=secondFactorLabels, qvalueMethod=qvalueMethod)
                        
                        pall_array.append(array)
                        cleanData.pop()

                        j_m += 1
                        start_1 = start_1 + block
                        end_1 = end_1 + block
                        if end_1 >= total_row_1:
                            end_1 = total_row_1
             
                        inner_bar.update(1)

                i_m += 1
                j_m = 0
                start_1 = 1
                end_1 = block
                if end_1 >= total_row_1:
                    end_1 = total_row_1

                start_0 = start_0 + block
                end_0 = end_0 + block
                if end_0 >= total_row_0:
                    end_0 = total_row_0

                outer_bar.update(1)

        data_set = np.vstack(pall_array)

        X = data_set[:,0].tolist()
        Y = data_set[:,1].tolist()
        lsaP = data_set[:,9]
        PCC = data_set[:,11]
        SCC = data_set[:,16]
        SPCC = data_set[:,13]
        SSCC = data_set[:,18]

        qvalues = qvalue_func(lsaP).tolist()
        pccqvalues = qvalue_func(PCC).tolist()
        sccqvalues = qvalue_func(SCC).tolist()
        spccqvalues = qvalue_func(SPCC).tolist()
        ssccqvalues = qvalue_func(SSCC).tolist()

        data_0 = np.column_stack((qvalues, pccqvalues, spccqvalues, sccqvalues, ssccqvalues, X, Y))
        data_set = np.hstack((data_set, data_0))
        
        merged_hf.create_dataset(f'data_{1}_{1}', data = data_set)

        dataFile.close()
        extraFile.close()

    print("finishing up...", file=sys.stderr)
    end_time=time.time()
    print("time elapsed %f seconds" % (end_time - start_time), file=sys.stderr)

if __name__=="__main__":
    main()
