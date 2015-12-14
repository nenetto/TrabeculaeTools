import numpy as np
import SimpleITK as sitk
import time
import os
from datetime import datetime, date, time, timedelta
import sys
import pandas
from PythonTools import io
from pyimagej.pyimagej import MacroImageJ
from scipy.spatial.distance import dice, jaccard, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule



def DownsampleImage(ImagePath, ResultPath, DownsamplingFactor = 3):
    '''Read image and create a downsampled version according to DownsamplingFactor
    '''
    # Declare starting time for process
    startTime = time()

    image = sitk.Cast(sitk.ReadImage(ImagePath), sitk.sitkFloat32)

    voxelSize = np.array(image.GetSpacing());
    imageSize = np.array(image.GetSize());

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

def PrintPercentage(percentage, preMessage = ''):
    '''This function only prints a percentage during processing
    '''
    global __previousTime
    global __previousPercentage

    try:
        __previousTime
        __previousPercentage
    except:
        __previousPercentage = 0.00001
        __previousTime = datetime.now()

    # Calculate percentage step
    percentageStep = percentage - __previousPercentage


    # Calculate time step from previous percentage step
    elapsedTime = datetime.now() - __previousTime

    # Calculate Remaining Time
    remainingTime = timedelta(seconds=elapsedTime.total_seconds() * (100.0 - percentage)/percentageStep)
    #remainingTime = str(remainingTime)
    remainingTime = "[" + str(remainingTime) + "]"


    nlines = (np.round(20 * percentage/100.0)).astype(int)
    newString = preMessage + "[{0:.2f}%]".format(percentage) + "--[" + '|'*nlines + ' '*(20-nlines) + "]-- Expected end in " + remainingTime
    if(percentage == 100.0):
        newString = newString + ' Finished!'
        __previousTime = datetime.now()
        __previousPercentage = 0.00001

    sys.stdout.write("\r%s" % newString)
    sys.stdout.flush()

def JoinCSVFiles(PathToFileList, PathToFileComplete):
    '''
        Join all files with the same pandas structure to a unique file.

        Keyword arguments:
        PathToFileList             -- List of files to be joined
        PathToFileComplete         -- Path where to save joined data
        PringDebug         [True]  -- True if printing messages are wanted
    '''
    # Declare starting time for process
    startTime = time()

    first = True
    for csvFile in PathToFileList:
        if first:
            TotalDataFrame = pandas.read_csv(csvFile,index_col=0)
            first = False
        else:
            newDataFrame = pandas.read_csv(csvFile,index_col=0)
            TotalDataFrame = TotalDataFrame.append(newDataFrame, ignore_index=True)


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
    startTime = time()

    # Set the random Seed
    if RandomSeed is not None:
        np.random.seed(RandomSeed)


    totalBoneMask = sitk.Cast(sitk.ReadImage(MaskFilePath), sitk.sitkUInt8)



    kernelEdge = (np.round(SizeRoImm / np.array(totalBoneMask.GetSpacing()))).astype(int)


    # Eroding mask
    voxelSize = np.array(totalBoneMask.GetSpacing());
    imageSize = np.array(totalBoneMask.GetSize());

    # Downsample image by a factor of 4

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

    itkEroder = sitk.BinaryErodeImageFilter()
    itkEroder.SetKernelType(sitk.sitkBox)
    itkEroder.SetKernelRadius(radius)
    itkEroder.SetForegroundValue(255.0)
    itkEroder.SetBackgroundValue(0.0)
    itkEroder.SetBoundaryToForeground(False)
    erodedtotalBoneMask = itkEroder.Execute(erodedtotalBoneMask)

    # Upsample image recovering space

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



    if nFoundRoIs == 0:
        print "[#####]    - Returning None because No RoIs were found for a size of {0:.2f}".format(SizeRoImm)
        return None

    if ErodedMaskFilePath is not None:


        sitk.WriteImage(erodedtotalBoneMask, ErodedMaskFilePath)

    if NRandomRoIs is None:
        if PrintDebug:
            print "[{0:.2f} s]".format(startTime) + "    - NRandomRoIs is [None] -> Extracting all possible RoIs"
            nRoIs = nFoundRoIs
    else:
        if( NRandomRoIs > 0 and NRandomRoIs <= nFoundRoIs):
            if PrintDebug:
                print "[{0:.2f} s]".format( startTime) + "    - Extracting {0:d} RoIs".format(NRandomRoIs)
            maxInt = sys.maxint - 1
            # Avoiding Overflow
            if (nFoundRoIs > maxInt):
                if PrintDebug:
                    print "[{0:.2f} s]".format( startTime) + "    - Correcting Overflow"
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
                print "[{0:.2f} s]".format( startTime) + "    - NRandomRoIs [{0:d}] > [{1:d}]Found RoIs Number".format(NRandomRoIs,nFoundRoIs)
                print "[{0:.2f} s]".format( startTime) + "        - Extracting {0:d} RoIs".format(nFoundRoIs)
            nRoIs = nFoundRoIs


    # Transform to world coordinates (mm)
    voxelSize = np.array(totalBoneMask.GetSpacing())

    Vx = Vx * voxelSize[0]
    Vy = Vy * voxelSize[1]
    Vz = Vz * voxelSize[2]

    # Creating RoI file structure
    # FileFrom (MaskFilePath), RoI Number, RoI Size mm, Center X mm, Center Y mm, Center Z mm,

    if PrintDebug:
                print "[{0:.2f} s]".format( startTime) + "    - Creating RoI Data Structure"


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
            print "[{0:.2f} s]".format( startTime) + "    - Saving RoIs to "
            print "[{0:.2f} s]".format( startTime) + "        - " + RoIcsvFilePath
        RoIStructure.to_csv(RoIcsvFilePath)

    if PrintDebug:
                print "[{0:.2f} s]".format( startTime) + "    - Finished"

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

    RoIStructure = JoinCSVFiles(PathToFileList,PathToFileComplete)
    PrintPercentage(100.0, preMessage = 'Creating RoI File: ')

def CreateRoI(ImageFilePath, RoIDefinition, RoIFilePath = None):
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
    startTime = time()


    # Read image


    image = sitk.Cast(sitk.ReadImage(ImageFilePath), sitk.sitkFloat32)
    imageSize, voxelSize = np.array(image.GetSize()), np.array(image.GetSpacing())

    # Creating Crop filter
    centerRoI = np.array([  RoIDefinition['Center x mm'],\
                            RoIDefinition['Center y mm'],\
                            RoIDefinition['Center z mm']])

    startFilterRoI = list(np.round( (centerRoI - RoIDefinition['RoI Size mm']/2.0) / voxelSize).astype(int))
    sizeFilterRoI  = list(np.round( RoIDefinition['RoI Size mm'] / voxelSize ).astype(int))


    # Create filter crop and apply
    itkExtractRoI = sitk.RegionOfInterestImageFilter()
    itkExtractRoI.SetIndex(startFilterRoI)
    itkExtractRoI.SetSize(sizeFilterRoI)
    imageRoI = itkExtractRoI.Execute(image)
    image = None

    if RoIFilePath is None:

        RoIFilePath = ImageFilePath[:-4] + '_{1:.2f}mm_RoI{0:d}.nii'.format(RoIDefinition['RoI Number'],RoIDefinition['RoI Size mm'])



    sitk.WriteImage(imageRoI, RoIFilePath)

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
    startTime = time()

    if PrintDebug:
        print "[{0:.2f} s]".format( startTime) + "    - Extracting RoI with parameters: "
        print RoIDefinition

    # Read image
    if PrintDebug:
        print "[{0:.2f} s]".format( startTime) + "    - Reading Image"

    image = sitk.Cast(sitk.ReadImage(ImageFilePath), sitk.sitkFloat32)
    imageSize, voxelSize = np.array(image.GetSize()), np.array(image.GetSpacing())

    # Calculating RoI Size safety in moving image space
    sizeRoImmModified = RoIDefinition['RoI Size mm'] * np.sqrt(2.0)

    # Transforming center of RoI to Moving Space
    if PrintDebug:
            print "[{0:.2f} s]".format( startTime) + "    - Transforming RoI center to moving space"
    TReg = io.load_tfm(TransformationFile)

    centerRoI = np.array([  RoIDefinition['Center x mm'],\
                            RoIDefinition['Center y mm'],\
                            RoIDefinition['Center z mm'],\
                            1.0])

    centroidTransformed = np.dot(TReg, centerRoI)[:3]

    startFilterRoI = list(np.round( (centroidTransformed - sizeRoImmModified/2.0) / voxelSize).astype(int))
    sizeFilterRoI  = list(np.round( sizeRoImmModified / voxelSize ).astype(int))
    if PrintDebug:
        print "[{0:.2f} s]".format( startTime) + "    - RoI starts at ", startFilterRoI
        print "[{0:.2f} s]".format( startTime) + "    - RoI size      ", sizeFilterRoI

    # Create filter crop and apply
    if PrintDebug:
            print "[{0:.2f} s]".format( startTime) + "    - Extracting RoI in Moving Space"
    itkExtractRoI = sitk.RegionOfInterestImageFilter()
    itkExtractRoI.SetIndex(startFilterRoI)
    itkExtractRoI.SetSize(sizeFilterRoI)
    imageRoI = itkExtractRoI.Execute(image)
    image = None

    # Register RoI
    if PrintDebug:
            print "[{0:.2f} s]".format( startTime) + "    - Transforming RoI to Target space"
    imageRoIReference = sitk.Cast(sitk.ReadImage(ReferenceRoIImageFilePath), sitk.sitkFloat32)

    # Resampling Reference RoI to imageRoI Spacing
    if PrintDebug:
            print "[{0:.2f} s]".format( startTime) + "    - Resampling..."
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
            print "[{0:.2f} s]".format( startTime) + "    - ..."
    T = sitk.ReadTransform(TransformationFile)
    imageRoI = sitk.Resample(imageRoI, imageRoIReference, T, sitk.sitkBSpline, sitk.sitkFloat32)

    if RoIFilePath is None:
        if PrintDebug:
            print "[{0:.2f} s]".format( startTime) + "    - No RoI file Name provided"
        RoIFilePath = ImageFilePath[:-4] + '_{1:.2f}mm_RoI{0:d}.nii'.format(RoIDefinition['RoI Number'],RoIDefinition['RoI Size mm'])

    if PrintDebug:
            print "[{0:.2f} s]".format( startTime) + "    - Saving RoI file to "
            print "[{0:.2f} s]".format( startTime) + "    - " + RoIFilePath

    sitk.WriteImage(imageRoI, RoIFilePath)
    if PrintDebug:
        print "[{0:.2f} s]".format( startTime) + "    - Finished"

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
    startTime = time()

    segmentTrabecula = MacroImageJ(imagejPath = imagejPath,\
                               macroPath = macroPath,\
                               xmlDefinition = xmlDefinition,\
                               defaultTimeout = defaultTimeout)

    segmentTrabecula.runMacro(  SMOOTH_Sigma = SMOOTH_Sigma,\
                                TH_Erosion = TH_Erosion,\
                                TH_Dilation = TH_Dilation,\
                                inputImage = PathToRoIfile,\
                                outputImage = PathToSegmentedRoIfile)


    #print "Finish!, total time:  {0:.2f}".format( startTime)

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
    startTime = time()


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

    print "Finish!, total time:  {0:.2f}".format( startTime)

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

def LocalThresholding(imagePath, GaussRadius, GradCutOffLow, GradCutOffHigh, BackGroundValue):

    original = sitk.Cast(sitk.ReadImage(imagePath), sitk.sitkFloat32)

    # Smooth image
    spacing = original.GetSpacing()[0]
    GaussSigma = spacing * GaussRadius
    smoothed = sitk.DiscreteGaussian(original,GaussSigma)

    # Calculate Gradient
    gradient = sitk.SobelEdgeDetection(smoothed)

    # Get image data
    gradientData = sitk.GetArrayFromImage(gradient)

    # Histogram
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    Nbins = 1000
    #n, bins, patches = ax.hist(gradientData.flatten(), Nbins, normed=True, facecolor='green', alpha=0.75)
    n, bins = np.histogram(gradientData.flatten(), bins=Nbins)

    # Cumulative Histogram
    ncum = np.cumsum(n)


    ncum = (1.0*ncum)/np.max(ncum)
    #l = ax.plot(bins[:-1], ncum*np.max(n), 'r-', linewidth=3)




    GradCutOffHighComputed = (bins[:-1])[ncum > GradCutOffHigh][0]
    GradCutOffLowComputed  = (bins[:-1])[ncum > GradCutOffLow][0]

    #ax.axvline(GradCutOffLowComputed, color='b', linestyle='-', linewidth=3)
    #ax.axvline(GradCutOffHighComputed, color='m', linestyle='-', linewidth=3)

    #ax.set_xlabel('Smarts')
    #ax.set_ylabel('Probability')
    #ax.set_title('Gradient Histogram')
    #ax.grid(True)
    #plt.show()

    # Calculate Edges image
    softedges = gradientData*0.0
    hardedges = gradientData*0.0
    noedges = np.ones_like(gradientData)

    hardedges[gradientData >= GradCutOffHighComputed] = 1
    noedges = noedges - hardedges
    hardedges = sitk.GetImageFromArray(hardedges, isVector=False)
    hardedges.SetOrigin(gradient.GetOrigin())
    hardedges.SetSpacing(gradient.GetSpacing())
    hardedges = sitk.Cast(hardedges, sitk.sitkUInt8)

    softedges[gradientData >= GradCutOffLowComputed] = 1
    softedges[gradientData >= GradCutOffHighComputed] = 0
    noedges = noedges - softedges
    softedges = sitk.GetImageFromArray(softedges, isVector=False)
    softedges.SetOrigin(gradient.GetOrigin())
    softedges.SetSpacing(gradient.GetSpacing())
    softedges = sitk.Cast(softedges, sitk.sitkUInt8)



    # Discard not-connected weak edges
    # We dilate the strong edges to mask the connected weak edges
    # Calculate Radius
    radius = 1 # one voxel closest

    # Create filter
    itkDilater = sitk.BinaryDilateImageFilter()
    itkDilater.SetKernelType(sitk.sitkBox)
    itkDilater.SetKernelRadius(radius)
    itkDilater.SetForegroundValue(1.0)
    itkDilater.SetBackgroundValue(0.0)
    itkDilater.SetBoundaryToForeground(False)
    hardedgesDilated = itkDilater.Execute(hardedges)

    # Multiply Soft edges
    softedges = softedges * hardedgesDilated

    # Set of edges
    setedges = sitk.GetArrayFromImage(hardedges)
    setedges = setedges + sitk.GetArrayFromImage(softedges) * (-1)

    setedges = sitk.GetImageFromArray(setedges, isVector=False)
    setedges.SetOrigin(gradient.GetOrigin())
    setedges.SetSpacing(gradient.GetSpacing())
    setedges = sitk.Cast(setedges, sitk.sitkInt8)

    hardedgesDilated = None
    hardedges = None
    softedges = None
    gradientData = None


    localThresholds = sitk.GetArrayFromImage(smoothed) * np.abs(sitk.GetArrayFromImage(setedges))
    localThresholds = sitk.GetImageFromArray(localThresholds, isVector=False)
    localThresholds.SetOrigin(original.GetOrigin())
    localThresholds.SetSpacing(original.GetSpacing())
    localThresholds = sitk.Cast(localThresholds, sitk.sitkFloat32)

    maskLocalThreshold = np.abs(sitk.GetArrayFromImage(setedges))
    maskLocalThreshold = sitk.GetImageFromArray(maskLocalThreshold, isVector=False)
    maskLocalThreshold.SetOrigin(original.GetOrigin())
    maskLocalThreshold.SetSpacing(original.GetSpacing())
    maskLocalThreshold = sitk.Cast(maskLocalThreshold, sitk.sitkInt8)

    # Dilate image Thresholds
    i = 0
    currentThr = localThresholds
    currentMask = maskLocalThreshold
    #localThresholds = sitk.GetArrayFromImage(localThresholds)
    while (sitk.GetArrayFromImage(currentThr).min() == 0):
        #print i, 'Dilating'
        i += 1

        # Dilation step for the Thresholds image
        currentThrDilate = sitk.GrayscaleDilate(currentThr,1,sitk.sitkBox)

        # Create mask dilating original by 1
        radius = 1 # one voxel closest
        itkDilater = sitk.BinaryDilateImageFilter()
        itkDilater.SetKernelType(sitk.sitkBox)
        itkDilater.SetKernelRadius(radius)
        itkDilater.SetForegroundValue(1.0)
        itkDilater.SetBackgroundValue(0.0)
        itkDilater.SetBoundaryToForeground(False)
        currentMaskDilated = itkDilater.Execute(currentMask)


        # Create diference mask and multiply for original
        diffMask = currentMaskDilated - currentMask
        currentMask = currentMaskDilated

        # Create next voxels to sum
        nextVoxelsToSum = sitk.Cast(diffMask, sitk.sitkFloat32)*currentThrDilate

        # Final Step adding new thresholds
        currentThr = currentThr + nextVoxelsToSum

    # For the Thrasholding adjustment we will need a gauss filtered image and an std filtering of the image
    thresholds = currentThr

    # Create the Gauss filtered image
    maxSizeKernel = 3
    thresholdsMean = sitk.DiscreteGaussian(thresholds, GaussSigma, maxSizeKernel)

    # Create the STD filtered image
    thresholdsSTD = sitk.Noise(thresholds,(1,1,1))


    # Get image matrix for

    originalForFilter = sitk.GetArrayFromImage(original)
    thresholds = sitk.GetArrayFromImage(thresholds)
    thresholdsMean = sitk.GetArrayFromImage(thresholdsMean)
    thresholdsSTD = sitk.GetArrayFromImage(thresholdsSTD)
    finalMask = originalForFilter*0.0

    # Threshold the image

    thresholdCondition = thresholdsMean - thresholdsSTD
    adjustedThreshold = thresholds * (1 + (thresholdsMean - thresholds)/(thresholdsMean - BackGroundValue) )

    # This is selected as Trabeculae
    finalMask[(originalForFilter > thresholds) & \
              (originalForFilter > thresholdCondition)] = 1.0

    finalMask[(originalForFilter > thresholds) & \
              (originalForFilter < thresholdCondition) & \
              (originalForFilter > adjustedThreshold)] = 1.0



    finalMask = sitk.GetImageFromArray(finalMask, isVector=False)
    finalMask.SetOrigin(original.GetOrigin())
    finalMask.SetSpacing(original.GetSpacing())
    finalMask = sitk.Cast(finalMask, sitk.sitkUInt8)

    #myshow3d(original, zslices=zslices2show, dpi=dpi2show,\
    #        title="Original")

    #myshow3d(finalMask, zslices=zslices2show, dpi=dpi2show,\
    #        title="Mask")

    return finalMask

def ResampleImage(targetImage, sourceImage):
    targetImage = sitk.Cast(sitk.ReadImage(targetImage), sitk.sitkFloat32)
    sourceImage = sitk.Cast(sitk.ReadImage(sourceImage), sitk.sitkFloat32)
    sourceImage = sitk.Resample(sourceImage , targetImage,sitk.Transform(), sitk.sitkBSpline)

    return sourceImage

def SegmentTrabeculaeLocalThresholding( imagePath, PathToSegmentedRoIfile, GaussRadius, GradCutOffLow, GradCutOffHigh, BackGroundValue ):
    '''
        Segment trabeculae from a RoI image and return segmentation parameters
        using BoneJ for segmentation

        Keyword arguments:
        imagePath              -- Image to be Segmented
        PathToSegmentedRoIfile -- Resulting Trabeculae image segmentation
    '''

    resultSegmentation = LocalThresholding(imagePath, GaussRadius, GradCutOffLow, GradCutOffHigh, BackGroundValue)
    resultSegmentation = resultSegmentation*255
    sitk.WriteImage(resultSegmentation,PathToSegmentedRoIfile)


    if os.path.isfile(PathToSegmentedRoIfile):
        ResultStruct = pandas.DataFrame(columns = [ 'Origin RoI',\
                                                'Segmented File',\
                                                'Segmentation Algorithm',\
                                                'GaussRadius',\
                                                'GradCutOffLow',\
                                                'GradCutOffHigh',\
                                                'BackGroundValue'\
                                                ], index = range(1))


        ResultStruct.iloc[0]  = pandas.Series({ 'Origin RoI' : imagePath,\
                                            'Segmented File' : PathToSegmentedRoIfile,\
                                            'Segmentation Algorithm' : 'Local Thresholding',\
                                            'GaussRadius' : GaussRadius,\
                                            'GradCutOffLow' : GradCutOffLow,\
                                            'GradCutOffHigh' : GradCutOffHigh,\
                                            'BackGroundValue' : BackGroundValue\
                                            })

        return ResultStruct
    else:
        print "\n[ERROR] Something was wrong segmenting image: " + PathToRoIfile.split('\\')[-1][:-4]
        return None

def SetImageOrigin(PathToRoIfile, PathToSegmentedRoIfile):
    original = sitk.Cast(sitk.ReadImage(PathToRoIfile), sitk.sitkFloat32)
    segmented = sitk.Cast(sitk.ReadImage(PathToSegmentedRoIfile), sitk.sitkUInt8)

    segmented.SetOrigin(original.GetOrigin())
    sitk.WriteImage(segmented,PathToSegmentedRoIfile)
