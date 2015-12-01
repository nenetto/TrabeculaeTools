##############################################################################################################################
#              PARAMETERS               ######################################################################################
##############################################################################################################################

NRandomRoIs = 2 # Number of Random Generated RoIs
NTransform = 2 # Number of Random Transformation Generated
sizeRoImmVector = [4.5, 5.5]# [5,6,7,3] # RoI size in mm that will be tested
RandomSeed = 1 # Random Seed for repetitivity
NSegmentationAlgorithms = 1 # In a future, this could let use more than one algorithm

# Evaluated Image path
originalImagePath = r'J:\Projects\JHUTrabeculae\Results\uCT_InterpolationEffect\Originals\uCT.nii'
# Total Bone mask for the evaluated Image path
boneMaskPath = r'J:\Projects\JHUTrabeculae\Results\uCT_InterpolationEffect\Originals\uCT_BoneMask.nii'

# Path to save results of analysis
pathToSaveDirResults = r'J:\Projects\JHUTrabeculae\Results\uCT_InterpolationEffect'

# Path to imageJ (Need BoneJ and NII reader plugins)
imagej_exe_path = r'J:\Projects\JHUTrabeculae\Software\Programs\Fiji\ImageJ-win64.exe'

# Macro definition for segmentation Trabeculae
fileXMLdescription = u'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\SegmentTrabeculaImageJMacro.xml'
macroPath = u'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\SegmentTrabeculaImageJMacro.ijm'
segmentTrabecula = macroImageJ(imagejPath = imagej_exe_path, macroPath = macroPath, xmlDefinition = fileXMLdescription)

# Macro definition for measure the Metrics
fileXMLdescription = u'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\BoneJMetricsImageJMacro.xml'
macroPath = u'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\BoneJMetricsImageJMacro.ijm'
BoneJMetrics = macroImageJ(imagejPath = imagej_exe_path, macroPath = macroPath, xmlDefinition = fileXMLdescription)

##############################################################################################################################
#              PROCESSING               ######################################################################################
##############################################################################################################################

# SavingDir:
#    .- GoldStandardRoIs: Random RoI images from original image
#        .- RoIMasks: image files of total bone mask eroded to fit size of RoI
#            .- BoneMaskImageName_RoIMask[RoISize]mm.nii
#        .- RandomRoIfiles: csv description for the randomly selected RoI
#            .- BoneMaskImageName_RandomRoIs[RoISize]mm.csv
#        .- RandomRoIImages: Generated RoI files from CSVs, the Gold Standard for comparison
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_GoldStandard.nii
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_GoldStandard_Alg[#SegmentationAlgorithm].tif
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_GoldStandard_Alg[#SegmentationAlgorithm]
#                .- data (results for the gold standard)
#                .- images (images results for the gold standard)
#
#    .- TransformedRoIs: Images that were transformed from Gold Standards
#        .- TransformationFiles: tfm files that describes the transformation applied
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_T[#Transformation].tfm
#        .- TransformedRoIs: Images that were transformed from Gold Standards
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_T[#Transformation].nii
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_T[#Transformation]_SegmentedTrabecula_Alg[#SegmentationAlgorithm].tif
#            .- ImageName_[RoISize]mm_RoINum[#RandomRoI]_T[#Transformation]_SegmentedTrabecula_Alg[#SegmentationAlgorithm]
#                .- data
#                .- images
#
#    .- MetricResults
#        .- Metrics_[RoISize]mm_RoINum[#RandomRoI]
#        .- Metrics_[RoISize]mm_RoINum[#RandomRoI]_T[#Transformation]
#        .-

# This could be run in parallel for each sizeRoImm.

totalProcess = time.time()
totalOfProcessings = len(sizeRoImmVector) * NRandomRoIs * NTransform * NSegmentationAlgorithms
currentProcess = 0


# Create directory where to save the metricResults
metricResultsPath = pathToSaveDirResults + '\MetricResults'
if not os.path.exists(metricResultsPath):
    os.makedirs(metricResultsPath)

# For each size RoI studied
for sizeRoImm in sizeRoImmVector:

    print "Current Analysis"
    print "    - Roi Size (mm):                    ", sizeRoImm
    print "    - Number of random RoIs:            ", NRandomRoIs
    print "    - Number of random Transformations: ", NTransform


    pathToSaveGoldStandardRoIImages = pathToSaveDirResults + '\\GoldStandardRoIs'
    if not os.path.exists(pathToSaveGoldStandardRoIImages):
        os.makedirs(pathToSaveGoldStandardRoIImages)


    RoIfileStructure = extractSquareRoIs( boneMaskPath, sizeRoImm, pathToSaveGoldStandardRoIImages, NRandomRoIs )

    # If none, a RoI of this size could not be found
    if(RoIfileStructure == None):
        print "[{0:.2f} s]".format(time.time() - totalProcess) + " RoI size is too big for fit inside mask"

    else:
        # For each Random generated RoI of this size
        for CurrentRoINumber in range(RoIfileStructure.shape[0]):

            # Get the Roi parameters
            currentRoIparameters = RoIfileStructure[CurrentRoINumber]

            # Define the filename for the Segmented image
            GoldStandarRoIimage = originalImagePath.split("\\")[-1][:-4] + '_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_GoldStandard.nii'

            pathToSaveGoldStandardRoIImages = pathToSaveDirResults + '\\GoldStandardRoIs\\RandomRoIImages'
            if not os.path.exists(pathToSaveGoldStandardRoIImages):
                os.makedirs(pathToSaveGoldStandardRoIImages)

            # Define the complete path for saving and create folder for saving results
            GoldStandarRoIimage = pathToSaveGoldStandardRoIImages + '\\' + GoldStandarRoIimage

            # Create RoI image Gold Standard
            maskImage(originalImagePath, currentRoIparameters, GoldStandarRoIimage)



            for SegmentationAlgorithm in range(NSegmentationAlgorithms):

                GoldStandarRoITrabeculaSegmentation = originalImagePath.split("\\")[-1][:-4] + '_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_GoldStandard_SegmentedTrabecula_Alg' + str(int(SegmentationAlgorithm))
                GoldStandarRoITrabeculaSegmentation = pathToSaveGoldStandardRoIImages + '\\' + GoldStandarRoITrabeculaSegmentation


                # Segment trabeculae using BoneJ algorithm
                if(SegmentationAlgorithm == 0 ):
                    GoldStandarRoITrabeculaSegmentation = GoldStandarRoITrabeculaSegmentation + '.nii'#'.tif'


                    segmentTrabecula.runMacro(SMOOTH_Sigma = 0.03,\
                                              TH_Erosion = 0,\
                                              TH_Dilation = 0,\
                                              inputImage = GoldStandarRoIimage,\
                                              outputImage = GoldStandarRoITrabeculaSegmentation)


                # Measure Metric for segmentation
                MetricsOutputDir = GoldStandarRoITrabeculaSegmentation[:-4:]
                if not os.path.exists(MetricsOutputDir):
                    os.makedirs(MetricsOutputDir)

                ## IMPORTANT make sure that centroid < radius. This is important for Anisotropy analysis of the current size RoI
                # ANISOTROPY_Radius < sizeRoImm
                pANISOTROPY_Radius = 0.9 * sizeRoImm/2.0;


                params = BoneJMetrics.runMacro( inputImage = GoldStandarRoITrabeculaSegmentation,\
                                                outputDir = MetricsOutputDir,\
                                                ANISOTROPY_Radius = pANISOTROPY_Radius)



                # Join all results under same file of results for this case
                resultsJoinedFile = metricResultsPath + '\\Metrics_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_Alg' + str(int(SegmentationAlgorithm)) + '.csv'

                joinBoneJResults(resultsPath = MetricsOutputDir, \
                                 resultsJoinedFile = resultsJoinedFile, \
                                 imageName = originalImagePath.split("\\")[-1][:-4],\
                                 roiName = 'RandomRoI', \
                                 roiSizeX = sizeRoImm, \
                                 roiSizeY = sizeRoImm, \
                                 roiSizeZ = sizeRoImm,\
                                 roINumber = CurrentRoINumber,\
                                 algorithm = SegmentationAlgorithm,\
                                 transf = -1,\
                                 goldStandard = 1,\
                                 parameters = params)


                # For each random RoI image generate random Transformations and use different segmentation algorithms
                pathToSaveTransforms = pathToSaveDirResults + '\\TransformedRoIs'
                if not os.path.exists(pathToSaveTransforms):
                    os.makedirs(pathToSaveTransforms)


                # For each Transformation
                for TNumber in range(NTransform):


                    #### [[[DOES NOT DEPENDS ON SEGMENTATION ALGORITHM]]]
                    if(SegmentationAlgorithm == 0 ):
                        # Transform the image
                        transformedImageName = originalImagePath.split("\\")[-1][:-4] + '_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_T' + str(int(TNumber))
                        transformedImage = randomTransformation(GoldStandarRoIimage,pathToSaveTransforms,transformedImageName)


                    transformedImageSegmentedTrabecula = transformedImage[:-4:] + '_SegmentedTrabecula_Alg' + str(int(SegmentationAlgorithm))

                    if(SegmentationAlgorithm == 0 ):
                        # Segment trabeculae using BoneJ algorithm
                        transformedImageSegmentedTrabecula =  transformedImageSegmentedTrabecula + '.nii'#'.tif'
                        segmentTrabecula.runMacro(SMOOTH_Sigma = 0.03,\
                                                  TH_Erosion = 0,\
                                                  TH_Dilation = 0,\
                                                  inputImage = transformedImage,\
                                                  outputImage = transformedImageSegmentedTrabecula)



                    # Measure Metric for this segmentation
                    MetricsOutputDir = transformedImageSegmentedTrabecula[:-4:]
                    if not os.path.exists(MetricsOutputDir):
                        os.makedirs(MetricsOutputDir)

                    ## IMPORTANT make sure that centroid < radius. This is important for Anisotropy analysis of the current size RoI
                    # ANISOTROPY_Radius < sizeRoImm
                    pANISOTROPY_Radius = 0.9 * sizeRoImm/2.0;


                    params = BoneJMetrics.runMacro( inputImage = transformedImageSegmentedTrabecula,\
                                                    outputDir = MetricsOutputDir,\
                                                    ANISOTROPY_Radius = pANISOTROPY_Radius)



                    # Join all results under same file of results for this case
                    resultsJoinedFile = metricResultsPath + '\Metrics_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_Alg' + str(int(SegmentationAlgorithm)) + '_T' + str(int(TNumber)) + '.csv'

                    joinBoneJResults(resultsPath = MetricsOutputDir, \
                                     resultsJoinedFile = resultsJoinedFile, \
                                     imageName = originalImagePath.split("\\")[-1][:-4],\
                                     roiName = 'RandomRoI', \
                                     roiSizeX = sizeRoImm, \
                                     roiSizeY = sizeRoImm, \
                                     roiSizeZ = sizeRoImm,
                                     roINumber = CurrentRoINumber,\
                                     algorithm = SegmentationAlgorithm,\
                                     transf = TNumber,\
                                     goldStandard = 0,\
                                     parameters = params)

                    #### [[[DEPENDS ON SEGMENTATION ALGORITHM]]]
                    ## Calculate Jaccard and Dice Index
                    similaritySavingFile = metricResultsPath + '\Similarity_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_Alg' + str(int(SegmentationAlgorithm)) + '_T' + str(int(TNumber)) + '.csv'
                    getSimilarityMetrics(   transformedImageSegmentedTrabecula, GoldStandarRoITrabeculaSegmentation,similaritySavingFile,\
                                            imageName = originalImagePath.split("\\")[-1][:-4],\
                                            roiName = 'RandomRoI', \
                                            roiSizeX = sizeRoImm, \
                                            roiSizeY = sizeRoImm, \
                                            roiSizeZ = sizeRoImm,
                                            roINumber = CurrentRoINumber,\
                                            algorithm = SegmentationAlgorithm,\
                                            transf = TNumber,\
                                            goldStandard = 0,\
                                            parameters = params)


                    currentProcess = currentProcess + 1.0
                    currentPercentage = (currentProcess/totalOfProcessings) * 100.0
                    print "[{0:.2f} s]".format(time.time() - totalProcess) + " - [",  currentPercentage,"] %"


print "TOTAL elapsed time:[{0:.2f} s]".format(time.time() - totalProcess)
