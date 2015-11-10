
########################################################################################
##                Study of the Interpolation Effect on the BoneJ metric               ##
########################################################################################
#
#    Author: Eugenio Marinetto (marinetto@jhu.edu, emarinetto@hggm.es) 11-2015
#
#    Description
#        .- This script launch a complete analysis in order to analize the effect of Interpolation over uCT images
#
# Parameters:
#        - NRandomRoIs: Number of random RoIs that will be generated for each combination of Parameters
#        - NTransform: Number of random transforms that will be applied to generate wrapped (interpolation affected) images
#        - sizeRoImmVector: List of studied RoI sizes over the images
#        - RandomSeed: Allow for repetitivity
#        - NSegmentationAlgorithms: [[FOR FUTURE]] number of SegmentationAlgorithms that will be studied. Currently only BoneJ Thresholding is supported.
#        - AlgorithmNameList: list of names for data matrix of the algorithms used for segmentation
#        - originalImagePath: Path to CT image that will be studied
#        - boneMaskPath: Complete bone mask for originalImagePath
#        - pathToSaveDirResults: Result data will be saved there
#        - imagej_exe_path: where is the exe for fiji. Att: BoneJ and Nifti Reader plugins must be installed
#        - SegmentTrabeculaImageJMacro: path to macro and xml description of the ImageJ Macro that will run the segmentation of BoneJ Thresholding which is the only supported
#        - BoneJMetricsImageJMacro: path to macro and xml description of the ImageJ MAcro that will run the Metri analysis
#
#
# pathToSaveDirResults:
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
########################################################################################
########################################################################################
########################################################################################

import sys
# Add Python Tools
pythonToolsDir = r'J:\Projects\JHUTrabeculae\Software\Python\PythonTools'
sys.path.append(pythonToolsDir)

# Add ctk-cli
ctkcli_DIR = r'J:\Projects\JHUTrabeculae\Software\Python\ctk-cli'
sys.path.append(ctkcli_DIR)

# Add TrabeculaeTools
TrabeculaeTools_DIR = r'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools'
sys.path.append(TrabeculaeTools_DIR)

from ImageJTools import macroImageJ
from ImageJTools.HelperBoneJ import joinBoneJResults, extractSquareRoIs, maskImage, randomTransformation, getSimilarityMetrics

from PythonTools import registration, transforms, transformations
from PythonTools.helpers import vtk as vtkhelper

import vtk
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import os
import time
import seaborn as sns


import seaborn as sns
import pandas as pd


########################################################################################
##                Change the parameters here for a new analysis                       ##
########################################################################################


NRandomRoIs = 20 # Number of Random Generated RoIs
NTransform = 20 # Number of Random Transformation Generated
sizeRoImmVector = list(np.arange(2.75,5.5,0.25))# [5,6,7,3] # RoI size in mm that will be tested
RandomSeed = 365214 # Random Seed for repetitivity
NSegmentationAlgorithms = 1 # In a future, this could let use more than one algorithm

# Evaluated Image path
originalImagePath = r'J:\Projects\JHUTrabeculae\Results\uCT_InterpolationEffect2\Originals\uCT.nii'
# Total Bone mask for the evaluated Image path
boneMaskPath = r'J:\Projects\JHUTrabeculae\Results\uCT_InterpolationEffect2\Originals\uCT_BoneMask.nii'

# Path to save results of analysis
pathToSaveDirResults = r'J:\Projects\JHUTrabeculae\Results\uCT_InterpolationEffectWeekend'

# Path to imageJ (Need BoneJ and NII reader plugins)
imagej_exe_path = r'J:\Projects\JHUTrabeculae\Software\Programs\Fiji\ImageJ-win64.exe'

# Macro definition for segmentation Trabeculae
fileXMLdescription = r'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\SegmentTrabeculaImageJMacro.xml'
macroPath = r'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\SegmentTrabeculaImageJMacro.ijm'
segmentTrabecula = macroImageJ(imagejPath = imagej_exe_path, macroPath = macroPath, xmlDefinition = fileXMLdescription)

# Macro definition for measure the Metrics
fileXMLdescription = r'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\BoneJMetricsImageJMacro.xml'
macroPath = r'J:\Projects\JHUTrabeculae\Software\Python\TrabeculaeTools\ImageJMacros\BoneJMetricsImageJMacro.ijm'
BoneJMetrics = macroImageJ(imagejPath = imagej_exe_path, macroPath = macroPath, xmlDefinition = fileXMLdescription)

# List of Algorithm Used for segmentation
AlgorithmNameList = ['BoneJ']


########################################################################################
##                               Analysis                                             ##
########################################################################################



# This control time information of the complete process
totalProcess = time.time()
totalOfProcessings = len(sizeRoImmVector) * NRandomRoIs * NTransform * NSegmentationAlgorithms
currentProcess = 0


# Create directory where to save the metricResults
metricResultsPath = pathToSaveDirResults + r'\MetricResults'
if not os.path.exists(metricResultsPath):
    os.makedirs(metricResultsPath)

# For each size RoI studied
for sizeRoImm in sizeRoImmVector:

    print "Current Analysis"
    print "    - Roi Size (mm):                    ", sizeRoImm
    print "    - Number of random RoIs:            ", NRandomRoIs
    print "    - Number of random Transformations: ", NTransform


    pathToSaveGoldStandardRoIImages = pathToSaveDirResults + r'\GoldStandardRoIs'
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
            GoldStandarRoIimage = originalImagePath.split('\\')[-1][:-4] + '_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_GoldStandard.nii'

            pathToSaveGoldStandardRoIImages = pathToSaveDirResults +r'\GoldStandardRoIs\RandomRoIImages'
            if not os.path.exists(pathToSaveGoldStandardRoIImages):
                os.makedirs(pathToSaveGoldStandardRoIImages)

            # Define the complete path for saving and create folder for saving results
            GoldStandarRoIimage = pathToSaveGoldStandardRoIImages +'\\' + GoldStandarRoIimage

            # Create RoI image Gold Standard
            maskImage(originalImagePath, currentRoIparameters, GoldStandarRoIimage)



            for SegmentationAlgorithm in range(NSegmentationAlgorithms):

                GoldStandarRoITrabeculaSegmentation = originalImagePath.split("\\")[-1][:-4] + '_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_GoldStandard_SegmentedTrabecula_Alg' + str(int(SegmentationAlgorithm))
                GoldStandarRoITrabeculaSegmentation = pathToSaveGoldStandardRoIImages +'\\' + GoldStandarRoITrabeculaSegmentation


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
                resultsJoinedFile = metricResultsPath +r'\Metrics_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_Alg' + str(int(SegmentationAlgorithm)) + '.csv'

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
                pathToSaveTransforms = pathToSaveDirResults +r'\TransformedRoIs'
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
                    resultsJoinedFile = metricResultsPath +r'\Metrics_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_Alg' + str(int(SegmentationAlgorithm)) + '_T' + str(int(TNumber)) + '.csv'

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
                    similaritySavingFile = metricResultsPath +r'\Similarity_' + str(currentRoIparameters[4]) + 'mm_RoINum' + str(int(CurrentRoINumber)) + '_Alg' + str(int(SegmentationAlgorithm)) + '_T' + str(int(TNumber)) + '.csv'
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



pathToResults=pathToSaveDirResults +r'\MetricResults'
pathToSavingFiles = pathToSaveDirResults +r'\SimilarityAnalysis'

# Create folder for resulting graphs
if not os.path.exists(pathToSavingFiles):
    os.makedirs(pathToSavingFiles)


# Create data structure for Metrics
metricsData = []

# Create data structure for Similarities
similarityData = []


MetricHeaderRead = False
SimilarityHeaderRead = False
for f in listdir(pathToResults):

    filePath = pathToResults + '\\' + f
    if 'Metrics' in f:
        #print 'Metric File    :' , f
        with open(filePath) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if spamreader.line_num == 1:
                    if (not MetricHeaderRead):
                        MetricHeaderRead = True
                        metricsData.append(row)
                else:
                    metricsData.append(row)

    elif 'Similarity' in f:
        #print 'Similarity File:', f
        with open(filePath) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if spamreader.line_num == 1:
                    if (not SimilarityHeaderRead):
                        SimilarityHeaderRead = True
                        similarityData.append(row)
                else:
                    similarityData.append(row)

## Prepare data for Metric Values
import pandas as pd
# Convert Numeric Values
for row in metricsData[1::]:
    for i in range(2,len(metricsData[0])):
        row[i] = float(row[i])

# Set grouping variables
for row in metricsData[1::]:

    for alg in range(len(AlgorithmNameList)):
        if row[10] == alg:
            row[10] = AlgorithmNameList[alg]
    if row[12] == 1:
        row[12] = 'GoldStandard'
    else:
        row[12] = 'Transformed'

# Fixing Names of variables for pandas
NamesTitles = []
for i in range(len(metricsData[0])):
    NamesTitles.append((metricsData[0][i]).replace('.', ' '))
    metricsData[0][i] = (metricsData[0][i]).replace('.', '')



# Prepare Data Frame
dfMetrics = pd.DataFrame(metricsData[1::], columns = metricsData[0])

# Set Category variables
dfMetrics["ImageName"] = dfMetrics["ImageName"].astype('category')
dfMetrics["RoIName"] = dfMetrics["RoIName"].astype('category')
dfMetrics["RoINumber"] = dfMetrics["RoINumber"].astype('category')
dfMetrics["Algorithm"] = dfMetrics["Algorithm"].astype('category')
dfMetrics["Transf"] = dfMetrics["Transf"].astype('category')
dfMetrics["GoldStandard"] = dfMetrics["GoldStandard"].astype('category')


# Make difference to Gold Standard
dfMetricsDiff = dfMetrics.copy()
for i in range(len(dfMetricsDiff)):
    if dfMetricsDiff.iloc[i].GoldStandard == 'Transformed':
        sub = dfMetricsDiff[   (dfMetricsDiff.GoldStandard == 'GoldStandard') &\
                        (dfMetricsDiff.RoISizeX == dfMetricsDiff.iloc[i].RoISizeX) & \
                        (dfMetricsDiff.RoISizeY == dfMetricsDiff.iloc[i].RoISizeY) & \
                        (dfMetricsDiff.RoISizeZ == dfMetricsDiff.iloc[i].RoISizeZ) & \
                        (dfMetricsDiff.RoINumber == dfMetricsDiff.iloc[i].RoINumber) & \
                        (dfMetricsDiff.Algorithm == dfMetricsDiff.iloc[i].Algorithm)]
        dfMetricsDiff.iloc[i,29::] =  dfMetricsDiff.iloc[i,29::] - sub.iloc[0,29::]

dfMetricsDiff = dfMetricsDiff[dfMetricsDiff.GoldStandard == 'Transformed']

## Prepare data for Metric Values

# Convert Numeric Values
for row in similarityData[1::]:
    for i in range(2,len(similarityData[0])):
        row[i] = float(row[i])

# Set grouping variables
for row in similarityData[1::]:
    for alg in range(len(AlgorithmNameList)):
        if row[10] == alg:
            row[10] = AlgorithmNameList[alg]
    if row[12] == 1:
        row[12] = 'GoldStandard'
    else:
        row[12] = 'Transformed'

# Fixing Names of variables for pandas
NamesTitles = []
for i in range(len(similarityData[0])):
    NamesTitles.append((similarityData[0][i]).replace('.', ' '))
    similarityData[0][i] = (similarityData[0][i]).replace('.', '')


# Prepare Data Frame
dfSimilarity = pd.DataFrame(similarityData[1::], columns = similarityData[0])

# Set Category variables
dfSimilarity["ImageName"] = dfSimilarity["ImageName"].astype('category')
dfSimilarity["RoIName"] = dfSimilarity["RoIName"].astype('category')
dfSimilarity["RoINumber"] = dfSimilarity["RoINumber"].astype('category')
dfSimilarity["Algorithm"] = dfSimilarity["Algorithm"].astype('category')
dfSimilarity["Transf"] = dfSimilarity["Transf"].astype('category')
dfSimilarity["GoldStandard"] = dfSimilarity["GoldStandard"].astype('category')

# Change RussellRao metric to inverse and melt data
dfSimilarity['Russellrao'] = 1.0 - dfSimilarity['Russellrao']
dfSimilarityMelt = pd.melt(dfSimilarity, id_vars=similarityData[0][0:29], value_vars=similarityData[0][29::],\
          var_name='SimilarityMetric', value_name='Similarity')


dfMetricsDiff.to_csv(pathToSavingFiles +r'\TotalMetricDiffData.csv')
dfSimilarityMelt.to_csv(pathToSavingFiles +r'\TotalSimilarityData.csv')


# Plot metrics

for variable in range(29,len(metricsData[0])):
    fileName = pathToSavingFiles +'\\' + metricsData[0][variable] + ".pdf"
    try:
        dfSub = dfMetricsDiff[np.isfinite(dfMetricsDiff[metricsData[0][variable]])]
        g = sns.factorplot(data = dfSub,\
                           x = 'RoISizeX',\
                           y=metricsData[0][variable],\
                           kind='violin',\
                           split=True,\
                           palette="muted",\
                           aspect=2)
        g.set(ylabel = 'Difference to Gold S.')
        g.set(title = metricsData[0][variable] +'\n')

        sns.stripplot(x="RoISizeX", y=metricsData[0][variable], data=dfSub,\
                      size=4, jitter=True, edgecolor="gray",color="white")


        g.savefig(fileName)

    except:
        print "Fail doing graph for: ", metricsData[0][variable]


# Plot Similarity


g = sns.factorplot(data = dfSimilarityMelt,\
                   x = 'SimilarityMetric',\
                   y='Similarity',\
                   kind='violin',\
                   split=True,\
                   palette="muted",\
                   aspect=1,\
                   size = 10,)
g.set(ylabel = 'Difference to Gold S.')
g.set(title = 'SimilarityMetric\n')

sns.stripplot(x="SimilarityMetric", y='Similarity', data=dfSimilarityMelt,\
              size=4, jitter=True, edgecolor="gray",color="white")

fileName = pathToSavingFiles +r'\SimilarityMetricsForSegmentations.pdf'
g.savefig(fileName)

JaccarDiceData = dfSimilarityMelt[(dfSimilarityMelt.SimilarityMetric == "Jaccard") | (dfSimilarityMelt.SimilarityMetric == "Dice")]
g = sns.factorplot(data = JaccarDiceData,\
                   x = 'SimilarityMetric',\
                   y='Similarity',\
                   kind='violin',\
                   split=True,\
                   palette="muted",\
                   aspect=1,\
                   size = 10,)
g.set(ylabel = 'Difference to Gold S.')
g.set(title = 'SimilarityMetric\n')

sns.stripplot(x="SimilarityMetric", y='Similarity', data=JaccarDiceData,\
              size=4, jitter=True, edgecolor="gray",color="white")

fileName = pathToSavingFiles +r'\SimilarityMetricsForSegmentations_JaccardDice.pdf'
g.savefig(fileName)


g = sns.factorplot(data = JaccarDiceData,\
                   x = 'RoISizeX',\
                   y='Similarity',\
                   kind='violin',\
                   hue='SimilarityMetric',\
                   split=True,\
                   palette="muted",\
                   aspect=1,\
                   size = 10,)
g.set(ylabel = 'Difference to Gold S.')
g.set(title = 'SimilarityMetric\n')

sns.stripplot(x="RoISizeX", y='Similarity', data=JaccarDiceData,\
              size=4, jitter=True, edgecolor="gray",color="white")

fileName = pathToSavingFiles +r'\SimilarityMetricsForSegmentations_JaccardDiceVsRoISize.pdf'
g.savefig(fileName)
