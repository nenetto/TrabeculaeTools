import numpy as np
import SimpleITK as sitk
import time
import os
import datetime
import sys
import pandas
from PythonTools import io
from pyimagej.pyimagej import MacroImageJ
from scipy.spatial.distance import dice, jaccard, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

def DownsampleImage(ImagePath, ResultPath, DownsamplingFactor = 3, PrintDebug = True):
    '''Read image and create a downsampled version according to DownsamplingFactor
    '''
    # Declare starting time for process
    startTime = time.time()

    image = sitk.Cast(sitk.ReadImage(ImagePath), sitk.sitkFloat32)

    voxelSize = np.array(image.GetSpacing());
    imageSize = np.array(image.GetSize());

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Downsampling..."
    downSampleFactor = DownsamplingFactor
    itkResampler = sitk.ResampleImageFilter()
    itkResampler.SetInterpolator(sitk.sitkLinear)
    itkResampler.SetDefaultPixelValue(0)
    itkResampler.SetTransform(sitk.Transform())
    itkResampler.SetOutputSpacing(voxelSize * downSampleFactor)
    itkResampler.SetSize(np.round(imageSize / downSampleFactor).astype(int))
    image = itkResampler.Execute(image)

    sitk.WriteImage(image, ResultPath)

    image = None
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Finished!"

def PrintPercentage(percentage, preMessage = ''):
    '''This function only prints a percentage during processing
    '''
    timing = datetime.datetime.now().strftime('%H:%M:%S')
    timing = "[" + timing + "]"

    nlines = (np.round(20 * percentage/100.0)).astype(int)
    newString = preMessage + timing + "--[" + '|'*nlines + ' '*(20-nlines) + "]"
    if(percentage == 100.0):
        newString = newString + ' Finished!'

    sys.stdout.write("\r%s" % newString)
    sys.stdout.flush()

def JoinCSVFiles(PathToFileList, PathToFileComplete, PrintDebug = True):
    '''
        Join all files with the same pandas structure to a unique file.

        Keyword arguments:
        PathToFileList             -- List of files to be joined
        PathToFileComplete         -- Path where to save joined data
        PringDebug         [True]  -- True if printing messages are wanted
    '''
    # Declare starting time for process
    startTime = time.time()

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Reading and joining files..."

    first = True
    for csvFile in PathToFileList:
        if first:
            TotalDataFrame = pandas.read_csv(csvFile,index_col=0)
            first = False
        else:
            newDataFrame = pandas.read_csv(csvFile,index_col=0)
            TotalDataFrame = TotalDataFrame.append(newDataFrame, ignore_index=True)


    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Saving to File "

    TotalDataFrame.to_csv(PathToFileComplete)

    return TotalDataFrame

def CreateRoIFile(MaskFilePath, ErodedMaskFilePath = None, RoIcsvFilePath = None, SizeRoImm = 5, NRandomRoIs = 10, RandomSeed = 0,PrintDebug = True):
    '''
        Read MaskFilePath, perform erosion and save the result into ErodedMaskFilePath.
        Using the eroded mask, it creates a file into RoIcsvFilePath where RoIs are defined by
        the cencer of the RoI, the RoI number and the Size in mm.
        RoIs are cubes of edge SizeRoImm.

        Notes: If NRandomRoIs is set to 'None', all possible RoIs inside Bone will be extracted and
        saved to RoIcsvFilePath. However, depending on the size of the RoI, the number could be very
        high and is not recommended because of time performance.

        Keyword arguments:
        MaskFilePath                  -- Total Bone Mask file UInt8 [0,255] (255 == Bone)
        ErodedMaskFilePath    [None]  -- Eroded Total Bone Mask file UInt8 [0,255] (255 == Bone)
        RoIcsvFilePath        [None]  -- File where to save RoIs description
        SizeRoImm             [5]     -- Edge of the cube RoI in mm
        NRandomRoIs           [10]    -- Number of Random RoIs to extract
        RandomSeed            [1]     -- Random Seed for repetible results
        PringDebug            [True]  -- True if printing messages are wanted
    '''

    # Declare starting time for process
    startTime = time.time()

    # Set the random Seed
    if RandomSeed is not None:
        np.random.seed(RandomSeed)

    # Read the MaskFile
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Reading MaskFile"

    totalBoneMask = sitk.Cast(sitk.ReadImage(MaskFilePath), sitk.sitkUInt8)

    # Define edge
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Calculating Kernel Edge size"

    kernelEdge = (np.round(SizeRoImm / np.array(totalBoneMask.GetSpacing()))).astype(int)

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Kernel Size [",kernelEdge,"] voxels"

    # Eroding mask
    voxelSize = np.array(totalBoneMask.GetSpacing());
    imageSize = np.array(totalBoneMask.GetSize());

    # Downsample image by a factor of 4
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Downsampling..."
    downSampleFactor = 4.0
    itkResampler = sitk.ResampleImageFilter()
    itkResampler.SetInterpolator(sitk.sitkNearestNeighbor)
    itkResampler.SetDefaultPixelValue(0)
    itkResampler.SetTransform(sitk.Transform())
    itkResampler.SetOutputSpacing(voxelSize * downSampleFactor)
    itkResampler.SetSize(np.round(imageSize / downSampleFactor).astype(int))
    erodedtotalBoneMask = itkResampler.Execute(totalBoneMask)

    # Calculate Radius
    radius = np.round(SizeRoImm / np.array(voxelSize * downSampleFactor * 2)).astype(int)

    # Create filter
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Eroding..."
    itkEroder = sitk.BinaryErodeImageFilter()
    itkEroder.SetKernelType(sitk.sitkBox)
    itkEroder.SetKernelRadius(radius)
    itkEroder.SetForegroundValue(255.0)
    itkEroder.SetBackgroundValue(0.0)
    itkEroder.SetBoundaryToForeground(False)
    erodedtotalBoneMask = itkEroder.Execute(erodedtotalBoneMask)

    # Upsample image recovering space
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Upsampling..."
    itkResampler = sitk.ResampleImageFilter()
    itkResampler.SetInterpolator(sitk.sitkNearestNeighbor)
    itkResampler.SetDefaultPixelValue(0)
    itkResampler.SetTransform(sitk.Transform())
    itkResampler.SetOutputSpacing(voxelSize)
    itkResampler.SetSize(imageSize)
    erodedtotalBoneMask = itkResampler.Execute(erodedtotalBoneMask)



    # Get the voxels where mask is not 0 and compute RoI parameters
    volume = sitk.GetArrayFromImage(erodedtotalBoneMask)

    # Extract the idex for the centroids of masks
    (Vx, Vy, Vz) = np.nonzero(volume)

    # Compute number of found RoIs
    nFoundRoIs = len(Vx)

    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Extracting RoIs, {0:d} RoIs were found ".format(nFoundRoIs)

    if nFoundRoIs == 0:
        print "[#####][{0:.2f} s]".format(time.time() - startTime) + "    - Returning None because No RoIs were found for a size of {0:.2f}".format(SizeRoImm)
        return None

    if ErodedMaskFilePath is not None:
        if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Saving Eroded Mask to "
            print "[{0:.2f} s]".format(time.time() - startTime) + "        - " + ErodedMaskFilePath

        sitk.WriteImage(erodedtotalBoneMask, ErodedMaskFilePath)

    if NRandomRoIs is None:
        if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - NRandomRoIs is [None] -> Extracting all possible RoIs"
            nRoIs = nFoundRoIs
    else:
        if( NRandomRoIs > 0 and NRandomRoIs <= nFoundRoIs):
            if PrintDebug:
                print "[{0:.2f} s]".format(time.time() - startTime) + "    - Extracting {0:d} RoIs".format(NRandomRoIs)
            maxInt = sys.maxint - 1
            # Avoiding Overflow
            if (nFoundRoIs > maxInt):
                if PrintDebug:
                    print "[{0:.2f} s]".format(time.time() - startTime) + "    - Correcting Overflow"
                greaterTimes = nFoundRoIs/maxInt
                randomSample = np.random.random_integers(0, maxInt, nRandomRoIs)
                for i in range(greaterTimes):
                    randomSample = randomSample.astype(long) + np.random.random_integers(0, maxInt, NRandomRoIs).astype(long)
            else:
                randomSample = np.random.random_integers(0, nFoundRoIs, NRandomRoIs)

            Vx = Vx[randomSample];
            Vy = Vy[randomSample];
            Vz = Vz[randomSample];


            nRoIs = NRandomRoIs
        else:
            if PrintDebug:
                print "[{0:.2f} s]".format(time.time() - startTime) + "    - NRandomRoIs [{0:d}] > [{1:d}]Found RoIs Number".format(NRandomRoIs,nFoundRoIs)
                print "[{0:.2f} s]".format(time.time() - startTime) + "        - Extracting {0:d} RoIs".format(nFoundRoIs)
            nRoIs = nFoundRoIs


    # Transform to world coordinates (mm)
    voxelSize = np.array(totalBoneMask.GetSpacing())

    Vx = Vx * voxelSize[0]
    Vy = Vy * voxelSize[1]
    Vz = Vz * voxelSize[2]

    # Creating RoI file structure
    # FileFrom (MaskFilePath), RoI Number, RoI Size mm, Center X mm, Center Y mm, Center Z mm,

    if PrintDebug:
                print "[{0:.2f} s]".format(time.time() - startTime) + "    - Creating RoI Data Structure"


    RoIStructure = pandas.DataFrame(columns = [ 'File', \
                                                'RoI Number', \
                                                'RoI Size mm', \
                                                'Center x mm', \
                                                'Center y mm', \
                                                'Center z mm'],
                                    index = range(nRoIs))


    for i in range(nRoIs):
        RoIStructure.iloc[i] = pandas.Series({  'File' : MaskFilePath,\
                                            'RoI Number' : i,\
                                            'RoI Size mm': SizeRoImm,\
                                            'Center x mm': Vz[i],\
                                            'Center y mm': Vy[i],\
                                            'Center z mm': Vx[i]    })


    if RoIcsvFilePath is not None:
        if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Saving RoIs to "
            print "[{0:.2f} s]".format(time.time() - startTime) + "        - " + RoIcsvFilePath
        RoIStructure.to_csv(RoIcsvFilePath)

    if PrintDebug:
                print "[{0:.2f} s]".format(time.time() - startTime) + "    - Finished"

    # Return Structure
    return RoIStructure

def CreateRoIfileStudy(MaskFilePath, RoIFolder, RoISizeVector, NRandomRoIs, RandomSeed = 0, PrintDebug = False):

    np.random.seed(RandomSeed)

    if not os.path.exists(RoIFolder):
        os.makedirs(RoIFolder)

    N = len(RoISizeVector)
    percentageIt = 0.0

    PrintPercentage(0.0, preMessage = 'Creating RoI File: ')
    for SizeRoImm in RoISizeVector:
        ErodedMaskFilePath = os.path.join(RoIFolder,'ErodedBoneMask_{0:2.2f}.nii'.format(SizeRoImm))
        RoIcsvFilePath = os.path.join(RoIFolder,'RoIFile_{0:2.2f}.csv'.format(SizeRoImm))

        A = CreateRoIFile(MaskFilePath, \
                      ErodedMaskFilePath = ErodedMaskFilePath, \
                      RoIcsvFilePath = RoIcsvFilePath, \
                      SizeRoImm = SizeRoImm, \
                      NRandomRoIs = NRandomRoIs, \
                      RandomSeed = None, \
                      PrintDebug = PrintDebug)
        percentageIt = percentageIt + 1.0
        percentage = 100.0 * percentageIt/float(N)
        PrintPercentage(percentage, preMessage = 'Creating RoI File: ')


    PathToFileComplete = os.path.join(RoIFolder,'RoIFileComplete.csv')

    PathToFileList = []
    for f in os.listdir(RoIFolder):
        if 'RoIFile_' in f:
            PathToFileList.append(os.path.join(RoIFolder,f))

    RoIStructure = JoinCSVFiles(PathToFileList,PathToFileComplete, PrintDebug = False)
    PrintPercentage(100.0, preMessage = 'Creating RoI File: ')

def CreateRoI(ImageFilePath, RoIDefinition, RoIFilePath = None, PrintDebug = True):
    '''
        Crop ImageFilePath according to RoIDefinition parameters. If the RoIFilePath is set, the RoI will be
        saved into file with name RoIFilePath.

        Note: of RoIFilePath os None, the file with the name: ImageFilePath_{RoI Size}mm_RoI{RoI Number}.nii

        Keyword arguments:
        ImageFilePath              -- File to be cropped
        RoIDefinition              -- RoI definition as pandas Data Frame
        RoIFilePath        [Spec]  -- File where to save RoIs image
        PringDebug         [True]  -- True if printing messages are wanted
    '''
    # Declare starting time for process
    startTime = time.time()

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Extracting RoI with parameters: "
        print RoIDefinition

    # Read image
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Reading Image"

    image = sitk.Cast(sitk.ReadImage(ImageFilePath), sitk.sitkFloat32)
    imageSize, voxelSize = np.array(image.GetSize()), np.array(image.GetSpacing())

    # Creating Crop filter
    centerRoI = np.array([  RoIDefinition['Center x mm'],\
                            RoIDefinition['Center y mm'],\
                            RoIDefinition['Center z mm']])

    startFilterRoI = list(np.round( (centerRoI - RoIDefinition['RoI Size mm']/2.0) / voxelSize).astype(int))
    sizeFilterRoI  = list(np.round( RoIDefinition['RoI Size mm'] / voxelSize ).astype(int))
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - RoI starts at ", startFilterRoI
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - RoI size      ", sizeFilterRoI

    # Create filter crop and apply
    itkExtractRoI = sitk.RegionOfInterestImageFilter()
    itkExtractRoI.SetIndex(startFilterRoI)
    itkExtractRoI.SetSize(sizeFilterRoI)
    imageRoI = itkExtractRoI.Execute(image)
    image = None

    if RoIFilePath is None:
        if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - No RoI file Name provided"
        RoIFilePath = ImageFilePath[:-4] + '_{1:.2f}mm_RoI{0:d}.nii'.format(RoIDefinition['RoI Number'],RoIDefinition['RoI Size mm'])

    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Saving RoI file to "
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - " + RoIFilePath

    sitk.WriteImage(imageRoI, RoIFilePath)
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Finished"

def CreateRoITransformed(ImageFilePath, RoIDefinition, TransformationFile, ReferenceRoIImageFilePath, RoIFilePath = None, PrintDebug = True):
    '''
        Crop ImageFilePath according to RoIDefinition parameters. It transforms the parameters to moving space
        (ImageFilePath), crop the RoI and transforms it back to Target space (ReferenceRoIImageFilePath)

        Note: of RoIFilePath os None, the file with the name: ImageFilePath_{RoI Size}mm_RoI{RoI Number}.nii

        Keyword arguments:
        ImageFilePath              -- File to be cropped
        RoIDefinition              -- RoI definition as pandas Data Frame
        TransformationFile         -- Transformation tfm file that transform Moving to Target
        ReferenceRoIImageFilePath  -- RoI file from the moving image in moving image space
        RoIFilePath        [Spec]  -- File where to save RoIs image
        PringDebug         [True]  -- True if printing messages are wanted
    '''

    # Declare starting time for process
    startTime = time.time()

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Extracting RoI with parameters: "
        print RoIDefinition

    # Read image
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Reading Image"

    image = sitk.Cast(sitk.ReadImage(ImageFilePath), sitk.sitkFloat32)
    imageSize, voxelSize = np.array(image.GetSize()), np.array(image.GetSpacing())

    # Calculating RoI Size safety in moving image space
    sizeRoImmModified = RoIDefinition['RoI Size mm'] * np.sqrt(2.0)

    # Transforming center of RoI to Moving Space
    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Transforming RoI center to moving space"
    TReg = io.load_tfm(TransformationFile)

    centerRoI = np.array([  RoIDefinition['Center x mm'],\
                            RoIDefinition['Center y mm'],\
                            RoIDefinition['Center z mm'],\
                            1.0])

    centroidTransformed = np.dot(TReg, centerRoI)[:3]

    startFilterRoI = list(np.round( (centroidTransformed - sizeRoImmModified/2.0) / voxelSize).astype(int))
    sizeFilterRoI  = list(np.round( sizeRoImmModified / voxelSize ).astype(int))
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - RoI starts at ", startFilterRoI
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - RoI size      ", sizeFilterRoI

    # Create filter crop and apply
    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Extracting RoI in Moving Space"
    itkExtractRoI = sitk.RegionOfInterestImageFilter()
    itkExtractRoI.SetIndex(startFilterRoI)
    itkExtractRoI.SetSize(sizeFilterRoI)
    imageRoI = itkExtractRoI.Execute(image)
    image = None

    # Register RoI
    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Transforming RoI to Target space"
    imageRoIReference = sitk.Cast(sitk.ReadImage(ReferenceRoIImageFilePath), sitk.sitkFloat32)

    # Resampling Reference RoI to imageRoI Spacing
    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Resampling..."
    itkResampler = sitk.ResampleImageFilter()
    itkResampler.SetInterpolator(sitk.sitkNearestNeighbor)
    itkResampler.SetOutputSpacing(imageRoI.GetSpacing())
    itkResampler.SetDefaultPixelValue(0)
    itkResampler.SetTransform(sitk.Transform())
    itkResampler.SetOutputOrigin(imageRoIReference.GetOrigin())
    newSize = - 2 + np.floor(np.array(imageRoIReference.GetSize()) *\
                             np.array(imageRoIReference.GetSpacing()) /\
                             np.array(imageRoI.GetSpacing())).astype(int)
    itkResampler.SetSize(list(newSize))
    imageRoIReference = itkResampler.Execute(imageRoIReference)

    # Transform RoI to target Space
    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - ..."
    T = sitk.ReadTransform(TransformationFile)
    imageRoI = sitk.Resample(imageRoI, imageRoIReference, T, sitk.sitkBSpline, sitk.sitkFloat32)

    if RoIFilePath is None:
        if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - No RoI file Name provided"
        RoIFilePath = ImageFilePath[:-4] + '_{1:.2f}mm_RoI{0:d}.nii'.format(RoIDefinition['RoI Number'],RoIDefinition['RoI Size mm'])

    if PrintDebug:
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - Saving RoI file to "
            print "[{0:.2f} s]".format(time.time() - startTime) + "    - " + RoIFilePath

    sitk.WriteImage(imageRoI, RoIFilePath)
    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Finished"

def SegmentTrabeculaeBoneJ( imagejPath, macroPath, xmlDefinition, PathToRoIfile, PathToSegmentedRoIfile, defaultTimeout = 120, SMOOTH_Sigma = 0.03, TH_Erosion = 0, TH_Dilation = 0):
    '''
        Segment trabeculae from a RoI image and return segmentation parameters
        using BoneJ for segmentation

        Keyword arguments:
        imagejPath             -- ImageJ executable path
        macroPath              -- ImageJ Macro that segments the image
        xmlDefinition          -- xmlDefinition for the ImageJ Macro parameters
        PathToRoIfile          -- Image to be Segmented
        PathToSegmentedRoIfile -- Resulting Trabeculae image segmentation
        defaultTimeout         -- Max time fot the segmentation allowed
        SMOOTH_Sigma           -- Sigma in mm for the gaussian blurring before segmenting
        TH_Erosion             -- Number of erosion after segmentation
        TH_Dilation            -- Number of dilation after segmentation
    '''

    # Declare starting time for process
    startTime = time.time()

    segmentTrabecula = MacroImageJ(imagejPath = imagejPath,\
                               macroPath = macroPath,\
                               xmlDefinition = xmlDefinition,\
                               defaultTimeout = defaultTimeout)

    segmentTrabecula.runMacro(  SMOOTH_Sigma = SMOOTH_Sigma,\
                                TH_Erosion = TH_Erosion,\
                                TH_Dilation = TH_Dilation,\
                                inputImage = PathToRoIfile,\
                                outputImage = PathToSegmentedRoIfile)

    #print "Finish!, total time:  {0:.2f}".format(time.time() - startTime)

    if os.path.isfile(PathToSegmentedRoIfile):
        ResultStruct = pandas.DataFrame(columns = [ 'Origin RoI',\
                                                'Segmented File',\
                                                'Segmentation Algorithm',\
                                                'Smooth Sigma',\
                                                'Number of Erosions',\
                                                'Number of Dilations',\
                                                ], index = range(1))


        ResultStruct.iloc[0]  = pandas.Series({ 'Origin RoI' : PathToRoIfile,\
                                            'Segmented File' : PathToSegmentedRoIfile,\
                                            'Segmentation Algorithm' : 'BoneJ',\
                                            'Smooth Sigma' : SMOOTH_Sigma,\
                                            'Number of Erosions' : TH_Erosion,\
                                            'Number of Dilations' : TH_Dilation})

        return ResultStruct
    else:
        print "\n[ERROR] Something was wrong segmenting image: " + PathToRoIfile.split('\\')[-1][:-4]
        return None

def GetResultsFromSegmentation(dataSegmentation, dataMetricsParameters):
    '''
        Get the dataFrame results from:
           - Segmentation (Origin RoI, Segmented File, Segmentation Algorithm, Segmentation Parameters...)
           - Metric Parameters, parameters for the Macro that measure the Metrics

        Collect all information and returns a well formed data frame with all information

        Keyword arguments:
        dataSegmentation         -- Segmentation params and information
        dataMetricsParameters    -- Metric Parameters including the outputDir where metrics were saved
    '''
    # Declare starting time for process
    startTime = time.time()


    # Read Metric Data Results
    resultsPathData = dataMetricsParameters.iloc[0]['outputDir'] + r'\data'
    ImageName = dataSegmentation.iloc[0]['Origin RoI'].split('\\')[-1][:-4]
    onlyfiles = [ f for f in os.listdir(resultsPathData) if (os.path.isfile(os.path.join(resultsPathData,f)) & (ImageName in f) ) ]

    for f in onlyfiles:
        df = pandas.read_csv(os.path.join(resultsPathData,f),index_col=0)

        if "Anisotropy" in f:
            DegreeOfAnisotropy = df.iloc[0]['DA']
            DegreeOfAnisotropyAlternative = df.iloc[0]['tDA']
            if DegreeOfAnisotropyAlternative == 'Infinity':
                DegreeOfAnisotropyAlternative = np.inf

        if "Connectivity" in f:
            EulerCharacteristic = df.iloc[0]['Euler ch.']
            EulerCharacteristicDelta = df.iloc[0]['?(?)']
            Connectivity = df.iloc[0]['Connectivity']
            ConnetivityDensity = df.iloc[0]['Conn.D (mm^-3)']

        if "SMI" in f:
            SMIPlus = df.iloc[0]['SMI+']
            SMIMinus = df.iloc[0]['SMI-']
            SMI = df.iloc[0]['SMI']

        if "Thickness" in f:
            TrabecularThicknessMean = df.iloc[0]['Tb.Th Mean (mm)']
            TrabecularThicknessStd = df.iloc[0]['Tb.Th Std Dev (mm)']
            TrabecularThicknessMax = df.iloc[0]['Tb.Th Max (mm)']
            TrabecularSpacingMean = df.iloc[0]['Tb.Sp Mean (mm)']
            TrabecularSpacingStd = df.iloc[0]['Tb.Sp Std Dev (mm)']
            TrabecularSpacingMax = df.iloc[0]['Tb.Sp Max (mm)']

        if "VolumeFraction_Surface" in f:
            BoneVolumemm3_S = df.iloc[0][1]
            TravecularVolumemm3_S = df.iloc[0][2]
            VolumeFraction_S = df.iloc[0][3]

        if "VolumeFraction_Voxel" in f:
            BoneVolumemm3_V = df.iloc[0][1]
            TravecularVolumemm3_V = df.iloc[0][2]
            VolumeFraction_V = df.iloc[0][3]


    # Create Data Frame
    dataMetrics = pandas.DataFrame(columns = [\
                                        'Degree Of Anisotropy', \
                                        'Alternative Degree Of Anisotropy', \
                                        'Euler Characteristic', \
                                        'Euler Characteristic Delta', \
                                        'Connectivity', \
                                        'Connetivity Density',\
                                        'SMI +',\
                                        'SMI -',\
                                        'SMI',\
                                        'Trabecular Thickness Mean',\
                                        'Trabecular Thickness Std',\
                                        'Trabecular Thickness Max',\
                                        'Trabecular Spacing Mean',\
                                        'Trabecular Spacing Std',\
                                        'Trabecular Spacing Max',\
                                        'Bone Volume mm3 Surface',\
                                        'Travecular Volumemm3 Surface',\
                                        'Volume Fraction Surface',\
                                        'Bone Volume mm3 Voxel',\
                                        'Travecular Volumemm3 Voxel',\
                                        'Volume Fraction Voxel'\
                                         ],
                                        index = range(1))
    # Populate Data frame
    dataMetrics.iloc[0] = pandas.Series({\
                                    'Degree Of Anisotropy' : DegreeOfAnisotropy, \
                                    'Alternative Degree Of Anisotropy' : DegreeOfAnisotropyAlternative, \
                                    'Euler Characteristic' : EulerCharacteristic, \
                                    'Euler Characteristic Delta' : EulerCharacteristicDelta, \
                                    'Connectivity' : Connectivity, \
                                    'Connetivity Density' : ConnetivityDensity,\
                                    'SMI +' : SMIPlus,\
                                    'SMI -' : SMIMinus,\
                                    'SMI' : SMI,\
                                    'Trabecular Thickness Mean' : TrabecularThicknessMean,\
                                    'Trabecular Thickness Std' : TrabecularThicknessStd,\
                                    'Trabecular Thickness Max' : TrabecularThicknessMax,\
                                    'Trabecular Spacing Mean' : TrabecularSpacingMean,\
                                    'Trabecular Spacing Std' : TrabecularSpacingStd,\
                                    'Trabecular Spacing Max' : TrabecularSpacingMax,\
                                    'Bone Volume mm3 Surface' : BoneVolumemm3_S,\
                                    'Travecular Volumemm3 Surface' : TravecularVolumemm3_S,\
                                    'Volume Fraction Surface' : VolumeFraction_S,\
                                    'Bone Volume mm3 Voxel' : BoneVolumemm3_V,\
                                    'Travecular Volumemm3 Voxel' : TravecularVolumemm3_V,\
                                    'Volume Fraction Voxel' : VolumeFraction_V\
                                    })



    # Delete inputImage and outputDir
    dataMetricsParametersFixed = dataMetricsParameters.drop('inputImage',1)
    dataMetricsParametersFixed = dataMetricsParametersFixed.drop('outputDir',1)

    # Join Together
    ResultingData = pandas.concat([dataSegmentation, dataMetricsParametersFixed, dataMetrics], axis=1, join_axes=[dataMetrics.index])

    print "Finish!, total time:  {0:.2f}".format(time.time() - startTime)

    return ResultingData

def GetResultsForComparison(resultsGoldStandard, resultsTested, RoISize, RoINumber, RoIX, RoIY, RoIZ):

    # Add the mask similarity to the structure
    goldStandardSegmentation = resultsGoldStandard.loc[0,'Segmented File']
    testedSegmentation = resultsTested.loc[0,'Segmented File']

    gsimage = sitk.Cast(sitk.ReadImage(goldStandardSegmentation), sitk.sitkFloat32)
    evimage = sitk.Cast(sitk.ReadImage(testedSegmentation), sitk.sitkFloat32)
    evimage = sitk.Resample(evimage , gsimage,sitk.Transform(), sitk.sitkNearestNeighbor, sitk.sitkUInt8)

    gsimage = sitk.GetArrayFromImage(gsimage)
    evimage = sitk.GetArrayFromImage(evimage)

    # ConvertMask to booleans

    gsimage = gsimage.astype(bool)
    evimage = evimage.astype(bool)


    similarityMetricsNames = [ 'Dice',\
                               'Jaccard',\
                               'Matching',\
                               'Rogerstanimoto',\
                               'Russellrao',\
                               'Sokalmichener',\
                               'Sokalsneath',\
                               'Yule',\
                               'RoI Size mm',\
                               'RoI Number',\
                               'Center x mm',\
                               'Center y mm',\
                               'Center z mm']

    similarityMetrics = np.array([  1.0 - dice(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - jaccard(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - matching(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - rogerstanimoto(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - russellrao(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - sokalmichener(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - sokalsneath(gsimage.flatten(), evimage.flatten()),\
                                    1.0 - yule(gsimage.flatten(), evimage.flatten()),\
                                    RoISize,\
                                    RoINumber,\
                                    RoIX,\
                                    RoIY,\
                                    RoIZ
                                    ])


    # Create Data Frame
    differenceDF = pandas.DataFrame(columns = [\
                                        'Relative Error Degree Of Anisotropy', \
                                        'Relative Error Alternative Degree Of Anisotropy', \
                                        'Relative Error Euler Characteristic', \
                                        'Relative Error Euler Characteristic Delta', \
                                        'Relative Error Connectivity', \
                                        'Relative Error Connetivity Density',\
                                        'Relative Error SMI +',\
                                        'Relative Error SMI -',\
                                        'Relative Error SMI',\
                                        'Relative Error Trabecular Thickness Mean',\
                                        'Relative Error Trabecular Thickness Std',\
                                        'Relative Error Trabecular Thickness Max',\
                                        'Relative Error Trabecular Spacing Mean',\
                                        'Relative Error Trabecular Spacing Std',\
                                        'Relative Error Trabecular Spacing Max',\
                                        'Relative Error Bone Volume mm3 Surface',\
                                        'Relative Error Travecular Volumemm3 Surface',\
                                        'Relative Error Volume Fraction Surface',\
                                        'Relative Error Bone Volume mm3 Voxel',\
                                        'Relative Error Travecular Volumemm3 Voxel',\
                                        'Relative Error Volume Fraction Voxel',\
                                        'Dice',\
                                        'Jaccard',\
                                        'Matching',\
                                        'Rogerstanimoto',\
                                        'Russellrao',\
                                        'Sokalmichener',\
                                        'Sokalsneath',\
                                        'Yule',\
                                        'RoI Size mm',\
                                        'RoI Number',\
                                        'Center x mm',\
                                        'Center y mm',\
                                        'Center z mm',\
                                        'Distance to zero',\
                                        'Image origin'\
                                         ],
                                        index = range(1))

    # Perform Difference and add to joined dataFrame
    DegreeOfAnisotropy = 100.0 * (resultsTested.loc[0,'Degree Of Anisotropy'] - resultsGoldStandard.loc[0,'Degree Of Anisotropy']) / resultsGoldStandard.loc[0,'Degree Of Anisotropy']
    DegreeOfAnisotropyAlternative = 100.0 * (resultsTested.loc[0,'Alternative Degree Of Anisotropy'] - resultsGoldStandard.loc[0,'Alternative Degree Of Anisotropy']) / resultsGoldStandard.loc[0,'Alternative Degree Of Anisotropy']
    EulerCharacteristic = 100.0 * (resultsTested.loc[0,'Euler Characteristic'] - resultsGoldStandard.loc[0,'Euler Characteristic']) / resultsGoldStandard.loc[0,'Degree Of Anisotropy']
    EulerCharacteristicDelta = 100.0 * (resultsTested.loc[0,'Euler Characteristic Delta'] - resultsGoldStandard.loc[0,'Euler Characteristic Delta']) / resultsGoldStandard.loc[0,'Euler Characteristic Delta']
    Connectivity = 100.0 * (resultsTested.loc[0,'Connectivity'] - resultsGoldStandard.loc[0,'Connectivity']) / resultsGoldStandard.loc[0,'Connectivity']
    ConnetivityDensity = 100.0 * (resultsTested.loc[0,'Connetivity Density'] - resultsGoldStandard.loc[0,'Connetivity Density']) / resultsGoldStandard.loc[0,'Connetivity Density']
    SMIPlus = 100.0 * (resultsTested.loc[0,'SMI +'] - resultsGoldStandard.loc[0,'SMI +']) / resultsGoldStandard.loc[0,'SMI +']
    SMIMinus = 100.0 * (resultsTested.loc[0,'SMI -'] - resultsGoldStandard.loc[0,'SMI -']) / resultsGoldStandard.loc[0,'SMI -']
    SMI = 100.0 * (resultsTested.loc[0,'SMI'] - resultsGoldStandard.loc[0,'SMI']) / resultsGoldStandard.loc[0,'SMI']
    TrabecularThicknessMean = 100.0 * (resultsTested.loc[0,'Trabecular Thickness Mean'] - resultsGoldStandard.loc[0,'Trabecular Thickness Mean']) / resultsGoldStandard.loc[0,'Trabecular Thickness Mean']
    TrabecularThicknessStd = 100.0 * (resultsTested.loc[0,'Trabecular Thickness Std'] - resultsGoldStandard.loc[0,'Trabecular Thickness Std']) / resultsGoldStandard.loc[0,'Trabecular Thickness Std']
    TrabecularThicknessMax = 100.0 * (resultsTested.loc[0,'Trabecular Thickness Max'] - resultsGoldStandard.loc[0,'Trabecular Thickness Max']) / resultsGoldStandard.loc[0,'Trabecular Thickness Max']
    TrabecularSpacingMean = 100.0 * (resultsTested.loc[0,'Trabecular Spacing Mean'] - resultsGoldStandard.loc[0,'Trabecular Spacing Mean']) / resultsGoldStandard.loc[0,'Trabecular Spacing Mean']
    TrabecularSpacingStd = 100.0 * (resultsTested.loc[0,'Trabecular Spacing Std'] - resultsGoldStandard.loc[0,'Trabecular Spacing Std']) / resultsGoldStandard.loc[0,'Trabecular Spacing Std']
    TrabecularSpacingMax = 100.0 * (resultsTested.loc[0,'Trabecular Spacing Max'] - resultsGoldStandard.loc[0,'Trabecular Spacing Max']) / resultsGoldStandard.loc[0,'Trabecular Spacing Max']
    BoneVolumemm3_S = 100.0 * (resultsTested.loc[0,'Bone Volume mm3 Surface'] - resultsGoldStandard.loc[0,'Bone Volume mm3 Surface']) / resultsGoldStandard.loc[0,'Bone Volume mm3 Surface']
    TravecularVolumemm3_S = 100.0 * (resultsTested.loc[0,'Travecular Volumemm3 Surface'] - resultsGoldStandard.loc[0,'Travecular Volumemm3 Surface']) / resultsGoldStandard.loc[0,'Travecular Volumemm3 Surface']
    VolumeFraction_S = 100.0 * (resultsTested.loc[0,'Volume Fraction Surface'] - resultsGoldStandard.loc[0,'Volume Fraction Surface']) / resultsGoldStandard.loc[0,'Volume Fraction Surface']
    BoneVolumemm3_V = 100.0 * (resultsTested.loc[0,'Bone Volume mm3 Voxel'] - resultsGoldStandard.loc[0,'Bone Volume mm3 Voxel']) / resultsGoldStandard.loc[0,'Bone Volume mm3 Voxel']
    TravecularVolumemm3_V = 100.0 * (resultsTested.loc[0,'Travecular Volumemm3 Voxel'] - resultsGoldStandard.loc[0,'Travecular Volumemm3 Voxel']) / resultsGoldStandard.loc[0,'Travecular Volumemm3 Voxel']
    VolumeFraction_V = 100.0 * (resultsTested.loc[0,'Volume Fraction Voxel'] - resultsGoldStandard.loc[0,'Volume Fraction Voxel']) / resultsGoldStandard.loc[0,'Volume Fraction Voxel']


    # Populate Data frame
    differenceDF.iloc[0] = pandas.Series({\
                                    'Relative Error Degree Of Anisotropy' : DegreeOfAnisotropy, \
                                    'Relative Error Alternative Degree Of Anisotropy' : DegreeOfAnisotropyAlternative, \
                                    'Relative Error Euler Characteristic' : EulerCharacteristic, \
                                    'Relative Error Euler Characteristic Delta' : EulerCharacteristicDelta, \
                                    'Relative Error Connectivity' : Connectivity, \
                                    'Relative Error Connetivity Density' : ConnetivityDensity,\
                                    'Relative Error SMI +' : SMIPlus,\
                                    'Relative Error SMI -' : SMIMinus,\
                                    'Relative Error SMI' : SMI,\
                                    'Relative Error Trabecular Thickness Mean' : TrabecularThicknessMean,\
                                    'Relative Error Trabecular Thickness Std' : TrabecularThicknessStd,\
                                    'Relative Error Trabecular Thickness Max' : TrabecularThicknessMax,\
                                    'Relative Error Trabecular Spacing Mean' : TrabecularSpacingMean,\
                                    'Relative Error Trabecular Spacing Std' : TrabecularSpacingStd,\
                                    'Relative Error Trabecular Spacing Max' : TrabecularSpacingMax,\
                                    'Relative Error Bone Volume mm3 Surface' : BoneVolumemm3_S,\
                                    'Relative Error Travecular Volumemm3 Surface' : TravecularVolumemm3_S,\
                                    'Relative Error Volume Fraction Surface' : VolumeFraction_S,\
                                    'Relative Error Bone Volume mm3 Voxel' : BoneVolumemm3_V,\
                                    'Relative Error Travecular Volumemm3 Voxel' : TravecularVolumemm3_V,\
                                    'Relative Error Volume Fraction Voxel' : VolumeFraction_V,\
                                    'Dice' : dice(gsimage.flatten(), evimage.flatten()),\
                                    'Jaccard' : jaccard(gsimage.flatten(), evimage.flatten()) ,\
                                    'Matching' : matching(gsimage.flatten(), evimage.flatten()),\
                                    'Rogerstanimoto' : rogerstanimoto(gsimage.flatten(), evimage.flatten()),\
                                    'Russellrao' : russellrao(gsimage.flatten(), evimage.flatten()),\
                                    'Sokalmichener' : sokalmichener(gsimage.flatten(), evimage.flatten()),\
                                    'Sokalsneath' : sokalsneath(gsimage.flatten(), evimage.flatten()),\
                                    'Yule' : yule(gsimage.flatten(), evimage.flatten()), \
                                    'RoI Size mm' : RoISize,\
                                    'RoI Number' : RoINumber,\
                                    'Center x mm' : RoIX,\
                                    'Center y mm' : RoIY,\
                                    'Center z mm' : RoIZ,\
                                    'Distance to zero' : np.sqrt(RoIX*RoIX + RoIY*RoIY + RoIZ*RoIZ),\
                                    'Image origin' : 'CBCT'
                                    })



    ResultingData = pandas.concat([resultsTested, differenceDF], axis=1, join_axes=[resultsTested.index])

    resultsGoldStandard['Image origin'] = 'uCT'
    resultsGoldStandard['RoI Size mm'] = RoISize
    resultsGoldStandard['RoI Number'] = RoINumber
    resultsGoldStandard['Center x mm'] = RoIX
    resultsGoldStandard['Center y mm'] = RoIY
    resultsGoldStandard['Center z mm'] = RoIZ
    resultsGoldStandard['Distance to zero'] = np.sqrt(RoIX*RoIX + RoIY*RoIY + RoIZ*RoIZ)

    ResultingData = ResultingData.append(resultsGoldStandard)

    return ResultingData
