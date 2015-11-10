
import csv
import numpy as np
from os import listdir
from os.path import isfile, join, exists
from numpy import genfromtxt
import vtk
from PythonTools.helpers import vtk as vtkhelper
from PythonTools import registration, transforms, transformations
from scipy import ndimage
from scipy.spatial.distance import dice, jaccard, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule
import math
import os
import time
import sys
import seaborn as sns
import pandas as pd

def joinBoneJResults(resultsPath, resultsJoinedFile, imageName = 'ImageName', roiName = 'ROIName', voxelSizeX = 1.0, voxelSizeY = 1.0, voxelSizeZ = 1.0, roiSizeX = 1.0, roiSizeY = 1.0, roiSizeZ = 1.0, roINumber = 0, algorithm = 0, transf = 0,goldStandard = 0, parameters = None):
    '''Join the results csv files after processing of a Segmented Image

    Keyword arguments:
    resultsPath      -- File CSV file path to save data (mandatory)
    imageName        -- Name of the image for CSV (default ImageName)
    roiName          -- Name of the RoI for the CSV (default ROIName)
    voxelSizeX [1.0] -- Voxel Size in X direction (mm) (default 1.0)
    voxelSizeY [1.0] -- Voxel Size in Y direction (mm) (default 1.0)
    voxelSizeZ [1.0] -- Voxel Size in Z direction (mm) (default 1.0)
    roiSizeX   [1.0] -- roi Size in X direction (voxels) (default 1.0)
    roiSizeY   [1.0] -- roi Size in Y direction (voxels) (default 1.0)
    roiSizeZ   [1.0] -- roi Size in Z direction (voxels) (default 1.0)
    roINumber  [0]   -- Number of Studied RoI (#) (default 0)
    algorithm  [0]   -- Segmentation Algorithm used (#) (default 0)
    transf     [0]   -- Transformation used (#) (default 0)
    goldStandard      -- If the image is a goldStandard (1) or not (0)

    '''



    roiVolume = voxelSizeX*roiSizeX * voxelSizeY*roiSizeY * voxelSizeZ*roiSizeZ

    # Check if resultsPath is a correct directory
    correctFlag = True
    print "[Info] Checking input result path..."
    if(exists(resultsPath)):
        print "    [OK] Path is correct"
        print "[Info] Looking for data and images folder..."
        if(exists(resultsPath + r'\data')):
            print "    [OK] Found data"
            pass
        else:
            print "    [Error] data folder NOT found"
            correctFlag = False
        if(exists(resultsPath + r'\images')):
            print "    [OK] Found images"
            pass
        else:
            print "    [Error] images folder NOT found"
            correctFlag = False
    else:
        "    [Error] Path is not correct"
        correctFlag = False

    if(not correctFlag):
        print '[Error] Aborting...'
        pass
    else:
        print '[Info] Checking data files for reading...'
        resultsPathData = resultsPath + r'\data'
        print '    [OK] data\\'
        onlyfiles = [ f for f in listdir(resultsPathData) if isfile(join(resultsPathData,f)) ]

        for f in onlyfiles:
            print '             :-' + f
            pass

        if(len(onlyfiles) == 12):
            correctFlag = True
            print '    [OK] Correct number of data files: ' + str(len(onlyfiles))
        else:
            correctFlag = False
            print '    [Error] Inorrect number of data files: ' + str(len(onlyfiles))

    if(not correctFlag):
        print '[Error] Aborting...'
        pass
    else:
        print '[Info] Reading files for further Analysis'

        for f in onlyfiles:
            path2f = join(resultsPathData,f)
            print '       ---------------------------------------------------------'
            print '       ' + f
            print '       ---------------------------------------------------------'

            if   "Anisotropy" in f:
                print '       [Info] Reading Anisotropy Results...'
                correctFlag = True
                columnNamesCorrected_Anisotropy = ['Anisotropy.Degree.of.Anisotropy',
                                        'Anisotropy.Degree.of.Anisotropy.Alternative',
                                        'Anisotropy.Fabric.Tensor.Vector.V11',
                                        'Anisotropy.Fabric.Tensor.Vector.V12',
                                        'Anisotropy.Fabric.Tensor.Vector.V13',
                                        'Anisotropy.Fabric.Tensor.Vector.V21',
                                        'Anisotropy.Fabric.Tensor.Vector.V22',
                                        'Anisotropy.Fabric.Tensor.Vector.V23',
                                        'Anisotropy.Fabric.Tensor.Vector.V31',
                                        'Anisotropy.Fabric.Tensor.Vector.V32',
                                        'Anisotropy.Fabric.Tensor.Vector.V33',
                                        'Anisotropy.Fabric.Tensor.Value.D1',
                                        'Anisotropy.Fabric.Tensor.Value.D2',
                                        'Anisotropy.Fabric.Tensor.Value.D3']
                my_data_Anisotropy = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,16))
                for i in range(0,len(columnNamesCorrected_Anisotropy)):
                    print '              -' + columnNamesCorrected_Anisotropy[i] + ': ' + str(my_data_Anisotropy[i])
                    pass

            elif "Connectivity" in f:
                print '       [Info] Reading Connectivity Results...'
                correctFlag = True
                columnNamesCorrected_Connectivity = ['Euler.Characteristic',
                                        'Connectivity.Euler.Characteristic.Delta',
                                        'Connectivity.Connectivity',
                                        'Connectivity.Connectivity.Density']
                my_data_Connectivity = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,6))
                for i in range(0,len(columnNamesCorrected_Connectivity)):
                    print '              -' + columnNamesCorrected_Connectivity[i] + ': ' + str(my_data_Connectivity[i])
                    pass

            elif "EllipsoidFactor_EF" in f:
                print '       [Info] Reading EllipsoidFactor EF Results...'
                correctFlag = True
                columnNamesCorrected_EllipsoidFactor_EF = ['Ellipsoid.Factor.EF.Area',
                                        'Ellipsoid.Factor.EF.Mean',
                                        'Ellipsoid.Factor.EF.StdDev',
                                        'Ellipsoid.Factor.EF.Mode',
                                        'Ellipsoid.Factor.EF.Min',
                                        'Ellipsoid.Factor.EF.Max',
                                        'Ellipsoid.Factor.EF.Circ',
                                        'Ellipsoid.Factor.EF.IntDen',
                                        'Ellipsoid.Factor.EF.Median',
                                        'Ellipsoid.Factor.EF.Skew',
                                        'Ellipsoid.Factor.EF.Kurt',
                                        'Ellipsoid.Factor.EF.Area.Percentage',
                                        'Ellipsoid.Factor.EF.RawIntDen',
                                        'Ellipsoid.Factor.EF.AR',
                                        'Ellipsoid.Factor.EF.Round',
                                        'Ellipsoid.Factor.EF.Solidity']
                my_data_EllipsoidFactor_EF = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,17))
                for i in range(0,len(columnNamesCorrected_EllipsoidFactor_EF)):
                    print '              -' + columnNamesCorrected_EllipsoidFactor_EF[i] + ': ' + str(my_data_EllipsoidFactor_EF[i])
                    pass

            elif "EllipsoidFactor_Max-ID" in f:
                print '       [Info] Reading EllipsoidFactor Max ID Results...'
                correctFlag = True
                columnNamesCorrected_EllipsoidFactor_Max_ID = ['Ellipsoid.Factor.Max.ID.Area',
                                        'Ellipsoid.Factor.Max.ID.Mean',
                                        'Ellipsoid.Factor.Max.ID.StdDev',
                                        'Ellipsoid.Factor.Max.ID.Mode',
                                        'Ellipsoid.Factor.Max.ID.Min',
                                        'Ellipsoid.Factor.Max.ID.Max',
                                        'Ellipsoid.Factor.Max.ID.Circ',
                                        'Ellipsoid.Factor.Max.ID.IntDen',
                                        'Ellipsoid.Factor.Max.ID.Median',
                                        'Ellipsoid.Factor.Max.ID.Skew',
                                        'Ellipsoid.Factor.Max.ID.Kurt',
                                        'Ellipsoid.Factor.Max.ID.Area.Percentage',
                                        'Ellipsoid.Factor.Max.ID.RawIntDen',
                                        'Ellipsoid.Factor.Max.ID.AR',
                                        'Ellipsoid.Factor.Max.ID.Round',
                                        'Ellipsoid.Factor.Max.ID.Solidity']
                my_data_EllipsoidFactor_Max_ID = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,17))
                for i in range(0,len(columnNamesCorrected_EllipsoidFactor_Max_ID)):
                    print '              -' + columnNamesCorrected_EllipsoidFactor_Max_ID[i] + ': ' + str(my_data_EllipsoidFactor_Max_ID[i])
                    pass

            elif "EllipsoidFactor_Mid_Long" in f:
                print '       [Info] Reading EllipsoidFactor Mid Long Results...'
                correctFlag = True
                columnNamesCorrected_EllipsoidFactor_Mid_Long = ['Ellipsoid.Factor.Mid.Long.Area',
                                        'Ellipsoid.Factor.Mid.Long.Mean',
                                        'Ellipsoid.Factor.Mid.Long.StdDev',
                                        'Ellipsoid.Factor.Mid.Long.Mode',
                                        'Ellipsoid.Factor.Mid.Long.Min',
                                        'Ellipsoid.Factor.Mid.Long.Max',
                                        'Ellipsoid.Factor.Mid.Long.Circ',
                                        'Ellipsoid.Factor.Mid.Long.IntDen',
                                        'Ellipsoid.Factor.Mid.Long.Median',
                                        'Ellipsoid.Factor.Mid.Long.Skew',
                                        'Ellipsoid.Factor.Mid.Long.Kurt',
                                        'Ellipsoid.Factor.Mid.Long.Area.Percentage',
                                        'Ellipsoid.Factor.Mid.Long.RawIntDen',
                                        'Ellipsoid.Factor.Mid.Long.AR',
                                        'Ellipsoid.Factor.Mid.Long.Round',
                                        'Ellipsoid.Factor.Mid.Long.Solidity']
                my_data_EllipsoidFactor_Mid_Long = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,17))
                for i in range(0,len(columnNamesCorrected_EllipsoidFactor_Mid_Long)):
                    print '              -' + columnNamesCorrected_EllipsoidFactor_Mid_Long[i] + ': ' + str(my_data_EllipsoidFactor_Mid_Long[i])
                    pass

            elif "EllipsoidFactor_ShortMid" in f:
                print '       [Info] Reading EllipsoidFactor ShortMid Results...'
                correctFlag = True
                columnNamesCorrected_EllipsoidFactor_ShortMid = ['Ellipsoid.Factor.Mid.Short.Area',
                                        'Ellipsoid.Factor.Mid.Short.Mean',
                                        'Ellipsoid.Factor.Mid.Short.StdDev',
                                        'Ellipsoid.Factor.Mid.Short.Mode',
                                        'Ellipsoid.Factor.Mid.Short.Min',
                                        'Ellipsoid.Factor.Mid.Short.Max',
                                        'Ellipsoid.Factor.Mid.Short.Circ',
                                        'Ellipsoid.Factor.Mid.Short.IntDen',
                                        'Ellipsoid.Factor.Mid.Short.Median',
                                        'Ellipsoid.Factor.Mid.Short.Skew',
                                        'Ellipsoid.Factor.Mid.Short.Kurt',
                                        'Ellipsoid.Factor.Mid.Short.Area.Percentage',
                                        'Ellipsoid.Factor.Mid.Short.RawIntDen',
                                        'Ellipsoid.Factor.Mid.Short.AR',
                                        'Ellipsoid.Factor.Mid.Short.Round',
                                        'Ellipsoid.Factor.Mid.Short.Solidity']
                my_data_EllipsoidFactor_ShortMid = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,17))
                for i in range(0,len(columnNamesCorrected_EllipsoidFactor_ShortMid)):
                    print '              -' + columnNamesCorrected_EllipsoidFactor_ShortMid[i] + ': ' + str(my_data_EllipsoidFactor_ShortMid[i])
                    pass


            elif "EllipsoidFactor_Volume" in f:
                print '       [Info] Reading EllipsoidFactor Volume Results...'
                correctFlag = True
                columnNamesCorrected_EllipsoidFactor_Volume = ['Ellipsoid.Factor.Volume.Area',
                                        'Ellipsoid.Factor.Volume.Mean',
                                        'Ellipsoid.Factor.Volume.StdDev',
                                        'Ellipsoid.Factor.Volume.Mode',
                                        'Ellipsoid.Factor.Volume.Min',
                                        'Ellipsoid.Factor.Volume.Max',
                                        'Ellipsoid.Factor.Volume.Circ',
                                        'Ellipsoid.Factor.Volume.IntDen',
                                        'Ellipsoid.Factor.Volume.Median',
                                        'Ellipsoid.Factor.Volume.Skew',
                                        'Ellipsoid.Factor.Volume.Kurt',
                                        'Ellipsoid.Factor.Volume.Area.Percentage',
                                        'Ellipsoid.Factor.Volume.RawIntDen',
                                        'Ellipsoid.Factor.Volume.AR',
                                        'Ellipsoid.Factor.Volume.Round',
                                        'Ellipsoid.Factor.Volume.Solidity']
                my_data_EllipsoidFactor_Volume = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,17))
                for i in range(0,len(columnNamesCorrected_EllipsoidFactor_Volume)):
                    print '              -' + columnNamesCorrected_EllipsoidFactor_Volume[i] + ': ' + str(my_data_EllipsoidFactor_Volume[i])
                    pass

            elif "FractalDimension" in f:
                print '       [Info] Reading Fractal Dimension Results...'
                correctFlag = True
                columnNamesCorrected_FractalDimension = ['Fractal.Dimension',
                                        'Fractal.Dimension.R.square']
                my_data_FractalDimension = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,4))
                for i in range(0,len(columnNamesCorrected_FractalDimension)):
                    print '              -' + columnNamesCorrected_FractalDimension[i] + ': ' + str(my_data_FractalDimension[i])
                    pass

            elif "SMI" in f:
                print '       [Info] Reading SMI Results...'
                correctFlag = True
                columnNamesCorrected_SMI = ['SMI.Concave',
                                        'SMI.Plus',
                                        'SMI.Minus',
                                        'SMI']
                my_data_SMI = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,6))
                for i in range(0,len(columnNamesCorrected_SMI)):
                    print '              -' + columnNamesCorrected_SMI[i] + ': ' + str(my_data_SMI[i])
                    pass

            elif "Thickness" in f:
                print '       [Info] Reading Thickness Results...'
                correctFlag = True
                columnNamesCorrected_Thickness = ['Thickness.Tb.Th.Mean',
                                        'Thickness.Tb.Th.StdDev',
                                        'Thickness.Tb.Th.Max',
                                        'Thickness.Tb.Sp.Mean',
                                        'Thickness.Tb.Th.StdDev',
                                        'Thickness.Tb.Th.Max']
                my_data_Thickness = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,8))
                for i in range(0,len(columnNamesCorrected_Thickness)):
                    print '              -' + columnNamesCorrected_Thickness[i] + ': ' + str(my_data_Thickness[i])
                    pass

            #'''
            #elif "TrabeculaeSkeleton_BranchInfo" in f:
            #    print '       [Info] Reading Trabeculae Skeleton Branch Info Results...'
            #    correctFlag = True
            #    columnNamesCorrected_TrabeculaeSkeleton_BranchInfo = ['Trabeculae.Skeleton.Branch.Skeleton.ID',
            #                            'Trabeculae.Skeleton.Branch.Branch.Length',
            #                            'Trabeculae.Skeleton.Branch.Extreme.origin.X',
            #                            'Trabeculae.Skeleton.Branch.Extreme.origin.Y',
            #                            'Trabeculae.Skeleton.Branch.Extreme.origin.Z',
            #                            'Trabeculae.Skeleton.Branch.Extreme.end.X',
            #                            'Trabeculae.Skeleton.Branch.Extreme.end.Y',
            #                            'Trabeculae.Skeleton.Branch.Extreme.end.Z',
            #                            'Trabeculae.Skeleton.Branch.Euclidean.Distance.Turtuosity']
            #    my_data_TrabeculaeSkeleton_BranchInfo = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,10))
            #    for i in range(0,len(columnNamesCorrected_TrabeculaeSkeleton_BranchInfo)):
            #        print '              -' + columnNamesCorrected_TrabeculaeSkeleton_BranchInfo[i] + ': ' + str(my_data_TrabeculaeSkeleton_BranchInfo[0,i])
            #        pass\

            #elif "TrabeculaeSkeleton" in f:
            #    print '       [Info] Reading Trabeculae Skeleton Results...'
            #    correctFlag = True
            #    columnNamesCorrected_TrabeculaeSkeleton = ['Trabeculae.Skeleton.Number.of.Branches',
            #                            'Trabeculae.Skeleton.Number.of.Junctions',
            #                            'Trabeculae.Skeleton.Number.of.EndPoint.Voxels',
            #                            'Trabeculae.Skeleton.Number.of.Junction.Voxels',
            #                            'Trabeculae.Skeleton.Number.of.Slab.Voxels',
            #                            'Trabeculae.Skeleton.Average.Branch.Length',
            #                            'Trabeculae.Skeleton.Number.of.Triple.Points',
            #                            'Trabeculae.Skeleton.Number.of.Cuadruple.Points',
            #                            'Trabeculae.Skeleton.Maximum.Branch.Length']
            #    my_data_TrabeculaeSkeleton = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(1,10))
            #    for i in range(0,len(columnNamesCorrected_TrabeculaeSkeleton)):
            #        print '              -' + columnNamesCorrected_TrabeculaeSkeleton[i] + ': ' + str(my_data_TrabeculaeSkeleton[0,i])
            #        pass

            #'''

            elif "VolumeFraction_Surface" in f:
                print '       [Info] Reading Volume Fraction Surface Results...'
                correctFlag = True
                columnNamesCorrected_VolumeFraction_Surface = ['Volume.Fraction.Surface.BV',
                                        'Volume.Fraction.Surface.TV',
                                        'Volume.Fraction.Surface.Volume.Fraction']
                my_data_VolumeFraction_Surface = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,5))
                for i in range(0,len(columnNamesCorrected_VolumeFraction_Surface)):
                    print '              -' + columnNamesCorrected_VolumeFraction_Surface[i] + ': ' + str(my_data_VolumeFraction_Surface[i])
                    pass

            elif "VolumeFraction_Voxel" in f:
                print '       [Info] Reading VolumeFraction Voxel Results...'
                correctFlag = True
                columnNamesCorrected_VolumeFraction_Voxel = ['Volume.Fraction.Voxel.BV',
                                        'Volume.Fraction.Voxel.TV',
                                        'Volume.Fraction.Voxel.Volume.Fraction']
                my_data_VolumeFraction_Voxel = genfromtxt(path2f, delimiter=',', skip_header=1, usecols = range(2,5))
                for i in range(0,len(columnNamesCorrected_VolumeFraction_Voxel)):
                    print '              -' + columnNamesCorrected_VolumeFraction_Voxel[i] + ': ' + str(my_data_VolumeFraction_Voxel[i])
                    pass

            else:
                correctFlag = False
                print '           [Error] File ' + f + ' is not recognised'


    if(not correctFlag):
        print '[Error] Aborting...'
        pass
    else:
        print '[Info] Joining files for further Analysis'


        columnNamesImage = ['Image.Name',
                            'RoI.Name',
                            'Voxel.Size.X',
                            'Voxel.Size.Y',
                            'Voxel.Size.Z',
                            'RoI.Size.X',
                            'RoI.Size.Y',
                            'RoI.Size.Z',
                            'RoI.Volume',
                            'RoI.Number',
                            'Algorithm',
                            'Transf',
                            'Gold.Standard']

        dataImage = np.array([  np.nan,
                                np.nan,
                                voxelSizeX,
                                voxelSizeY,
                                voxelSizeZ,
                                roiSizeX,
                                roiSizeY,
                                roiSizeZ,
                                roiVolume,
                                roINumber,
                                algorithm,
                                transf,
                                goldStandard])


        # Add parameters to the file
        for p in parameters:

            if not isinstance(parameters[p], basestring):
                columnNamesImage.append(p.replace('_','.'))
                dataImage = np.concatenate((dataImage,np.array([parameters[p]])), axis = 0)

        # Prepare common header
        columnNamesTotal =   columnNamesImage + \
                        columnNamesCorrected_Anisotropy + \
                        columnNamesCorrected_Connectivity + \
                        columnNamesCorrected_EllipsoidFactor_EF + \
                        columnNamesCorrected_EllipsoidFactor_Max_ID + \
                        columnNamesCorrected_EllipsoidFactor_Mid_Long + \
                        columnNamesCorrected_EllipsoidFactor_ShortMid + \
                        columnNamesCorrected_EllipsoidFactor_Volume + \
                        columnNamesCorrected_FractalDimension + \
                        columnNamesCorrected_SMI + \
                        columnNamesCorrected_Thickness + \
                        columnNamesCorrected_VolumeFraction_Surface + \
                        columnNamesCorrected_VolumeFraction_Voxel


        commonData = np.concatenate((   dataImage , \
                                        my_data_Anisotropy , \
                                        my_data_Connectivity , \
                                        my_data_EllipsoidFactor_EF , \
                                        my_data_EllipsoidFactor_Max_ID , \
                                        my_data_EllipsoidFactor_Mid_Long , \
                                        my_data_EllipsoidFactor_ShortMid , \
                                        my_data_EllipsoidFactor_Volume , \
                                        my_data_FractalDimension , \
                                        my_data_SMI , \
                                        my_data_Thickness , \
                                        my_data_VolumeFraction_Surface , \
                                        my_data_VolumeFraction_Voxel), \
                                    axis=0)


        print '[Info] Number of columns is ' + str(len(columnNamesTotal))

        '''
        # Check wich data has more rows
        maxRowNumber = my_data_TrabeculaeSkeleton_BranchInfo.shape[0] * my_data_TrabeculaeSkeleton.shape[0]

        print '[Info] Number of rows will be ' + str(maxRowNumber)

        # Create data structure
        print '[Info] Filling Data Matrix of [' + str(maxRowNumber) + ',' + str(len(columnNamesTotal)) + ']'

        try:
            totalData = np.zeros((maxRowNumber, len(columnNamesTotal)))
            iteratorData = 0
            for i in range(0,my_data_TrabeculaeSkeleton_BranchInfo.shape[0]):
                for j in range(0,my_data_TrabeculaeSkeleton.shape[0]):
                    totalData[iteratorData,:] = np.concatenate((  commonData ,
                                                my_data_TrabeculaeSkeleton_BranchInfo[i,:], \
                                                my_data_TrabeculaeSkeleton[j,:]), \
                                            axis=0)
                    iteratorData = iteratorData + 1
        except:
            totalData = commonData
        '''
        totalData = commonData



    if(not correctFlag):
        print '[Error] Aborting...'
        pass
    else:
        print '[Info] Savind data for Analysis'

    # Total Data
        #totalData
        # Need to set the Image Name and RoI Name
    # Complete Header
        #columnNamesTotal

        f = open(resultsJoinedFile, "w")
        # Write header
        line2write = ",".join(columnNamesTotal) + '\n'
        f.write(line2write)

        if len(totalData.shape) > 1:
            for i in range(0,totalData.shape[0]):
                line2write = imageName + ',' + roiName
                for e in totalData[i,2:]:
                    line2write = line2write + ',' + str(e)
                line2write = line2write + '\n'
                f.write(line2write)

        else:
            line2write = imageName + ',' + roiName
            for e in totalData[2:]:
                line2write = line2write + ',' + str(e)
            line2write = line2write + '\n'
            f.write(line2write)
        f.close

def extractSquareRoIs( pathToMaskFile, sizeRoImm, pathToSaveDir, nRandomRoIs ):

    print "#########################################################"
    print "               Extracting Square RoIs"
    print "#########################################################"

    start = time.time()

    if not os.path.exists(pathToSaveDir):
        os.makedirs(pathToSaveDir)

    # Load image matrix
    print "[{0:.2f} s]".format(time.time() - start) + "    - Reading mask..."
    vtkReader = vtk.vtkNIFTIImageReader()
    vtkReader.SetFileName(pathToMaskFile)
    vtkReader.Update()
    voxelSize = np.array(vtkReader.GetOutput().GetSpacing());
    SizeImage = np.array(vtkReader.GetOutput().GetExtent());
    SizeImage = SizeImage[[1,3,5]]

    # Get matrix data
    #mask = vtkhelper.ImageData_to_array(vtkReader.GetOutput())

    # Calculate size of the RoI in voxels
    roiSizeVoxels = np.round(sizeRoImm / voxelSize)

    for i in range(3):
        if(roiSizeVoxels[i]/2 == 0):
            roiSizeVoxels[i] += 1;


    print "[{0:.2f} s]".format(time.time() - start) + "    - Calculating RoI voxel size: ", roiSizeVoxels;

    # Resample image mask
    print "[{0:.2f} s]".format(time.time() - start) + "    - Downsampling mask..."
    vtkReslicer = vtk.vtkImageReslice()
    vtkReslicer.SetInputConnection( vtkReader.GetOutputPort() )
    newSpacing =  roiSizeVoxels * np.array([vtkReader.GetOutput().GetSpacing()[0],\
                                                 vtkReader.GetOutput().GetSpacing()[1],\
                                                 vtkReader.GetOutput().GetSpacing()[2]])
    vtkReslicer.SetOutputSpacing( newSpacing[0], newSpacing[1], newSpacing[2])
    vtkReslicer.SetInterpolationModeToNearestNeighbor()
    vtkReslicer.Update()


    # Eroding mask
    # Get matrix data
    maskDownsampled = vtkhelper.ImageData_to_array(vtkReslicer.GetOutput())
    # erode using 1voxel mask
    maskDownsampled = ndimage.morphology.binary_erosion(maskDownsampled)

    maskDownsampled = maskDownsampled.astype(int)


    print "[{0:.2f} s]".format(time.time() - start) + "    - Eroding mask..."
    #vtkEroder = vtk.vtkImageContinuousErode3D();
    #vtkEroder.SetInputConnection(vtkReslicer.GetOutputPort())
    #vtkEroder.SetKernelSize( 6,6,6); # 3 to ensure inside and x2 for the half roiSizeVoxels
    #vtkEroder.Update()


    # Resample image mask
    print "[{0:.2f} s]".format(time.time() - start) + "    - Upsampling mask..."
    vtkReslicer = vtk.vtkImageReslice()
    vtkReslicer.SetInputData( vtkhelper.ImageData_from_array(maskDownsampled))
    vtkReslicer.SetOutputExtent(vtkReader.GetOutput().GetExtent())
    newSpacing =  np.array([vtkReader.GetOutput().GetSpacing()[0],\
                                                 vtkReader.GetOutput().GetSpacing()[1],\
                                                 vtkReader.GetOutput().GetSpacing()[2]])
    vtkReslicer.SetOutputSpacing( newSpacing[0], newSpacing[1], newSpacing[2])
    vtkReslicer.SetInterpolationModeToNearestNeighbor()
    vtkReslicer.Update()
    erodedMask = vtkhelper.ImageData_to_array(vtkReslicer.GetOutput())
     # Set voxel size

    # Saving new mask
    # Folder for RoI results
    saveDir = pathToSaveDir + '\\RoIMasks'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)


    erodedImagePath = saveDir + '\\' + pathToMaskFile.split("\\")[-1][:-4] + '_RoIMask' + str(sizeRoImm) + 'mm.nii'
    print "[{0:.2f} s]".format(time.time() - start) + "    - Saving result mask..."
    vtkWriter = vtk.vtkNIFTIImageWriter()
    vtkWriter.SetFileName(erodedImagePath)
    vtkWriter.SetInputConnection(vtkReslicer.GetOutputPort())
    vtkWriter.Write()
    vtkWriter.Update()

    # Look for RoI centers
    print "[{0:.2f} s]".format(time.time() - start) + "    - Looking for RoI centers..."
    # Extract the idex for the centers
    (Vx, Vy, Vz) = np.nonzero(erodedMask)
    nFoundRoIs = len(Vx)

    # Select Randoms
    if( nFoundRoIs > 0):
        print "[{0:.2f} s]".format(time.time() - start) + "    - Found [" +  str(nFoundRoIs) + "] RoIs inside mask"

        if( nRandomRoIs > 0 and nRandomRoIs <= nFoundRoIs):
            print "[{0:.2f} s]".format(time.time() - start) + "    - Selecting a random sample of " + str(nRandomRoIs) + " RoIs"

            maxInt = sys.maxint - 1
            # Avoiding Overflow
            if (nFoundRoIs > maxInt):
                greaterTimes = nFoundRoIs/maxInt
                randomSample = np.random.random_integers(0, maxInt, nRandomRoIs)
                for i in range(greaterTimes):
                    randomSample = randomSample.astype(long) + np.random.random_integers(0, maxInt, nRandomRoIs).astype(long)
            else:
                randomSample = np.random.random_integers(0, nFoundRoIs, nRandomRoIs)

            Vx = Vx[randomSample];
            Vy = Vy[randomSample];
            Vz = Vz[randomSample];

            nRoIs = nRandomRoIs
        else:
            nRoIs = nFoundRoIs

    else:
        print "[{0:.2f} s]".format(time.time() - start) + "    - [##]: No RoIs of size " + str(sizeRoImm) + " mm were found inside the volume!"
        return None

    # Creating RoI file structure

    RoIfileStructure = np.zeros((nRoIs,9))

    for i in range(nRoIs):
        # Last element is the shape, 0: 'cube shape'
        RoIfileStructure[i,:] = np.array([i, Vx[i], Vy[i], Vz[i], sizeRoImm, 0 , Vx[i]*voxelSize[0], Vy[i]*voxelSize[1], (SizeImage[2] - Vz[i])*voxelSize[2] ])

    # Writing the RoI file
    print "[{0:.2f} s]".format(time.time() - start) + "    - Writing RoI file into " + 'RoIs_' + str(sizeRoImm) + 'mm.csv'
    saveDir = pathToSaveDir + '\\RandomRoIfiles'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    fileName = saveDir + '\\' + pathToMaskFile.split("\\")[-1][:-4] + '_RandomRoIs' + str(sizeRoImm) + 'mm.cvs'
    RoIfileStructureHeader = 'RoI.Number, Center.x, Center.y, Center.z, RoI.Size.mm, RoI.Shape, Center.x.mm, Center.y.mm, Center.z.mm'
    np.savetxt(fileName, \
               RoIfileStructure, \
               fmt='%10.5f', \
               delimiter=',', \
               newline='\n', \
               header=RoIfileStructureHeader, \
               comments='')

    print "[{0:.2f} s]".format(time.time() - start) + "    - Finished "
    return RoIfileStructure

def maskImage(imagePath, RoIparameters, roiImageFilepath):
    print "#########################################################"
    print "               Generating RoI from Image"
    print "#########################################################"
    start = time.time()
    # Read RoI Parameters
    print "[{0:.2f} s]".format(time.time() - start) + "    - RoI parameters:"
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI #             " , RoIparameters[0]
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI Center voxel  " , RoIparameters[1:4] , ' voxel'
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI Size (mm)     " , RoIparameters[4]
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI Center (mm)     " , RoIparameters[6:9]
    if (RoIparameters[5] == 0):
        print "        - RoI Shape          Cuboid"
        pass

    # Load image matrix
    print "[{0:.2f} s]".format(time.time() - start) + "    - Reading image..."

    vtkReader = vtk.vtkNIFTIImageReader()
    vtkReader.SetFileName(imagePath)
    vtkReader.Update()
    voxelSize = np.array(vtkReader.GetOutput().GetSpacing());

    # Get matrix data
    image = vtkhelper.ImageData_to_array(vtkReader.GetOutput())

    print "image size ", image.shape, "voxelSize ", voxelSize

    # Calculating limits of RoI
    roiSizeVoxels = np.round(RoIparameters[4] / voxelSize)
    print "[{0:.2f} s]".format(time.time() - start) + "    - Calculating RoI voxel size: ", roiSizeVoxels;
    for i in range(3):
        if(roiSizeVoxels[i]/2 == 0):
            roiSizeVoxels[i] += 1;


    roiSizeVoxels = (roiSizeVoxels - 1)/2
    limits = np.zeros((6))
    limits[0] = RoIparameters[1] - roiSizeVoxels[0]
    limits[1] = RoIparameters[1] + roiSizeVoxels[0]
    limits[2] = RoIparameters[2] - roiSizeVoxels[1]
    limits[3] = RoIparameters[2] + roiSizeVoxels[1]
    limits[4] = RoIparameters[3] - roiSizeVoxels[2]
    limits[5] = RoIparameters[3] + roiSizeVoxels[2]

    #try:

    # Creating mask
    print "[{0:.2f} s]".format(time.time() - start) + "    - Creating mask with limits ", limits, " in image of size ", image.shape
    roi = image[limits[0]:limits[1], limits[2]:limits[3],limits[4]:limits[5]]

    # Saving the mask
    print "[{0:.2f} s]".format(time.time() - start) + "    - Saving RoI file to " + roiImageFilepath

    roi = vtkhelper.ImageData_from_array(roi)
    # Set voxel size
    roi.SetSpacing(voxelSize[0], voxelSize[1], voxelSize[2])


    vtkWriter = vtk.vtkNIFTIImageWriter()
    vtkWriter.SetFileName(roiImageFilepath)
    vtkWriter.SetInputData(roi)
    vtkWriter.Write()
    vtkWriter.Update()

    print "[{0:.2f} s]".format(time.time() - start) + "    - Finished "

    #except:
    #    print "    - [##] Something was wrong creating the mask. Check the RoI parameters."

    print "#########################################################"

def randomTransformation(roiImageFilepath,pathToSaveDir, imagePathToSave):
    print "#########################################################"
    print "               Random transformation"
    print "#########################################################"
    start = time.time()
    # Read image for transforming
    vtkReader = vtk.vtkNIFTIImageReader()
    vtkReader.SetFileName(roiImageFilepath)
    vtkReader.Update()

    # Calculate maximum Traslation for random transformation
    maxTraslation = np.array(vtkReader.GetOutput().GetSpacing())[0]

    # Create random Angles and Transformation
    randomAngles = (np.random.random(3) - 0.5) * (math.pi/2.0) * np.array([1,1,1])
    randomTrans  =  (np.random.random(3) - 0.5) * 2 * maxTraslation

    # Convert to Transformation matrix
    randomT = transformations.compose_matrix(angles = randomAngles,\
                                                  translate = randomTrans)
    # print information
    print "[{0:.2f} s]".format(time.time() - start) + 'Random angles     : ', (np.random.random(3) - 0.5) * 360, 'degrees'
    print "[{0:.2f} s]".format(time.time() - start) + 'Random translation: ', randomTrans, ' mm'


    # Read image for transforming
    vtkReader = vtk.vtkNIFTIImageReader()
    vtkReader.SetFileName(roiImageFilepath)
    vtkReader.Update()

    # Create vtk Transform objects
    vtkRandomT = vtkhelper.vtkTransform_from_array(randomT)
    vtkRandomTinv = vtkhelper.vtkTransform_from_array(transformations.inverse_matrix(randomT))


    # Save transform
    print "[{0:.2f} s]".format(time.time() - start) + "    - Writing Transform file into " + roiImageFilepath.split("\\")[-1][:-4]  + '.tfm'
    pathToSaveTransforms = pathToSaveDir + '\\TransformationFiles'
    if not os.path.exists(pathToSaveTransforms):
        os.makedirs(pathToSaveTransforms)

    fileName = pathToSaveTransforms + '\\' + imagePathToSave + '.tfm'
    np.savetxt(fileName, \
               randomT, \
               fmt='%10.5f', \
               delimiter=',', \
               newline='\n', \
               header='', \
               comments='')



    # Apply transform
    vtkReslicer = vtk.vtkImageReslice()
    vtkReslicer.SetInputConnection(vtkReader.GetOutputPort())
    vtkReslicer.SetResliceTransform(vtkRandomT)
    vtkReslicer.SetInterpolationModeToCubic()
    vtkReslicer.SetOutputSpacing(   vtkReader.GetOutput().GetSpacing()[0],\
                                    vtkReader.GetOutput().GetSpacing()[1],\
                                    vtkReader.GetOutput().GetSpacing()[2]);
    vtkReslicer.SetOutputExtent(np.array(vtkReader.GetOutput().GetExtent()) * 8);
    print np.array(vtkReader.GetOutput().GetExtent()) * 8


    vtkReslicer2 = vtk.vtkImageReslice()
    vtkReslicer2.SetInputConnection(vtkReslicer.GetOutputPort())
    vtkReslicer2.SetResliceTransform(vtkRandomTinv)
    vtkReslicer2.SetInterpolationModeToCubic()
    vtkReslicer2.SetOutputSpacing(   vtkReader.GetOutput().GetSpacing()[0],\
                                     vtkReader.GetOutput().GetSpacing()[1],\
                                     vtkReader.GetOutput().GetSpacing()[2]);
    vtkReslicer2.SetOutputExtent(np.array(vtkReader.GetOutput().GetExtent()));

    # Saving wrapped image
    print "[{0:.2f} s]".format(time.time() - start) + "    - Writing Transform file into " + roiImageFilepath.split("\\")[-1][:-4] + '.nii'
    pathToSaveTransformedImages = pathToSaveDir + '\\TransformedFiles'
    if not os.path.exists(pathToSaveTransformedImages):
        os.makedirs(pathToSaveTransformedImages)

    roiImageFilepathResult = pathToSaveTransformedImages + "\\" + imagePathToSave + '.nii'

    vtkWriter = vtk.vtkNIFTIImageWriter()
    vtkWriter.SetFileName(roiImageFilepathResult)
    vtkWriter.SetInputConnection(vtkReslicer2.GetOutputPort())
    vtkWriter.Write()
    vtkWriter.Update()

    return roiImageFilepathResult

def getSimilarityMetrics(gsimageFile, evimageFile,resultsFile, imageName = 'ImageName', roiName = 'ROIName', voxelSizeX = 1.0, voxelSizeY = 1.0, voxelSizeZ = 1.0, roiSizeX = 1.0, roiSizeY = 1.0, roiSizeZ = 1.0, roINumber = 0, algorithm = 0, transf = 0,goldStandard = 0, parameters = None):
    print "Getting Similarity Metrics"

    roiVolume = voxelSizeX*roiSizeX * voxelSizeY*roiSizeY * voxelSizeZ*roiSizeZ

    vtkReaderGS = vtk.vtkNIFTIImageReader()
    vtkReaderGS.SetFileName(gsimageFile)
    vtkReaderGS.Update()
    voxelSizeGS = np.array(vtkReaderGS.GetOutput().GetSpacing())

    vtkReaderEV = vtk.vtkNIFTIImageReader()
    vtkReaderEV.SetFileName(evimageFile)
    vtkReaderEV.Update()
    voxelSizeEV = np.array(vtkReaderEV.GetOutput().GetSpacing())

    # Find the smaller spacing
    #print "Voxel Size Evaluated", voxelSizeEV
    #print "Voxel Size Gold Standard", voxelSizeGS
    if(voxelSizeEV[0] < voxelSizeGS[0]):
        #print 'Upsampling Evaluated image'
        vtkReslicer = vtk.vtkImageReslice()
        vtkReslicer.SetInputConnection( vtkReaderEV.GetOutputPort() )
        vtkReslicer.SetOutputSpacing( voxelSizeGS[0], voxelSizeGS[1], voxelSizeGS[2])
        vtkReslicer.SetInterpolationModeToNearestNeighbor()
        vtkReslicer.Update()

        gsimage = vtkhelper.ImageData_to_array(vtkReaderGS.GetOutput())
        evimage = vtkhelper.ImageData_to_array(vtkReslicer.GetOutput())

    elif(voxelSizeEV[0] < voxelSizeGS[0]):
        #print 'Upsampling Gold Standard image'
        vtkReslicer = vtk.vtkImageReslice()
        vtkReslicer.SetInputConnection( vtkReaderGS.GetOutputPort() )
        vtkReslicer.SetOutputSpacing( vtkReaderEV[0], vtkReaderEV[1], vtkReaderEV[2])
        vtkReslicer.SetInterpolationModeToNearestNeighbor()
        vtkReslicer.Update()

        gsimage = vtkhelper.ImageData_to_array(vtkReslicer.GetOutput())
        evimage = vtkhelper.ImageData_to_array(vtkReaderEV.GetOutput())

    else:
        #print 'Same Voxel Size'

        gsimage = vtkhelper.ImageData_to_array(vtkReaderGS.GetOutput())
        evimage = vtkhelper.ImageData_to_array(vtkReaderEV.GetOutput())

    # ConvertMask to booleans

    gsimage = gsimage.astype(bool)
    evimage = evimage.astype(bool)

    columnNamesImage = ['Image.Name',\
                        'RoI.Name',\
                        'Voxel.Size.X',\
                        'Voxel.Size.Y',\
                        'Voxel.Size.Z',\
                        'RoI.Size.X',\
                        'RoI.Size.Y',\
                        'RoI.Size.Z',\
                        'RoI.Volume',\
                        'RoI.Number',\
                        'Algorithm',\
                        'Transf',\
                        'Gold.Standard']


    dataImage = np.array([  np.nan,\
                            np.nan,\
                            voxelSizeX,\
                            voxelSizeY,\
                            voxelSizeZ,\
                            roiSizeX,\
                            roiSizeY,\
                            roiSizeZ,\
                            roiVolume,\
                            roINumber,\
                            algorithm,\
                            transf,\
                            goldStandard])


    # Add parameters to the file
    for p in parameters:

        if not isinstance(parameters[p], basestring):
            columnNamesImage.append(p.replace('_','.'))
            dataImage = np.concatenate((dataImage,np.array([parameters[p]])), axis = 0)

    similarityMetricsNames = [ 'Dice',\
                               'Jaccard',\
                               'Matching',\
                               'Rogerstanimoto',\
                               'Russellrao',\
                               'Sokalmichener',\
                               'Sokalsneath',\
                               'Yule']
    similarityMetrics = np.array([  dice(gsimage.flatten(), evimage.flatten()),\
                                    jaccard(gsimage.flatten(), evimage.flatten()),\
                                    matching(gsimage.flatten(), evimage.flatten()),\
                                    rogerstanimoto(gsimage.flatten(), evimage.flatten()),\
                                    russellrao(gsimage.flatten(), evimage.flatten()),\
                                    sokalmichener(gsimage.flatten(), evimage.flatten()),\
                                    sokalsneath(gsimage.flatten(), evimage.flatten()),\
                                    yule(gsimage.flatten(), evimage.flatten())])


    # Prepare common header
    columnNamesTotal =   columnNamesImage + similarityMetricsNames
    totalData = np.concatenate((dataImage , similarityMetrics), axis=0)

    f = open(resultsFile, "w")
    # Write header
    line2write = ",".join(columnNamesTotal) + '\n'
    f.write(line2write)
    line2write = imageName + ',' + roiName
    for e in totalData[2:]:
        line2write = line2write + ',' + str(e)
    line2write = line2write + '\n'
    f.write(line2write)
    f.close




    pass



###############################################################################################

###############################################################################################

def maskImageReg(imagePath, RoIparameters, roiImageFilepath, registrationFile):
    print "#########################################################"
    print "               Generating RoI from Image"
    print "#########################################################"
    start = time.time()
    # Read RoI Parameters
    print "[{0:.2f} s]".format(time.time() - start) + "    - RoI parameters:"
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI #             " , RoIparameters[0]
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI Center voxel  " , RoIparameters[1:4] , ' voxel'
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI Size (mm)     " , RoIparameters[4]
    print "[{0:.2f} s]".format(time.time() - start) + "        - RoI Center (mm)     " , RoIparameters[6:9]
    if (RoIparameters[5] == 0):
        print "        - RoI Shape          Cuboid"
        pass


    # Transform centroid of mask to Moving space
    TReg = io.load_tfm(registrationFile)
    InvTReg = np.linalg.inv(TReg)
    TransformedCentroid = np.dot(RoIparameters[6:9],InvTReg[0:3,0:3]) + InvTReg[0:3,3]


    # Load image matrix
    print "[{0:.2f} s]".format(time.time() - start) + "    - Reading image..."

    vtkReader = vtk.vtkNIFTIImageReader()
    vtkReader.SetFileName(imagePath)
    vtkReader.Update()
    voxelSize = np.array(vtkReader.GetOutput().GetSpacing());

    # Get matrix data
    image = vtkhelper.ImageData_to_array(vtkReader.GetOutput())

    print "image size ", image.shape, "voxelSize ", voxelSize

    # Calculate limits for cropping

    # Calculating limits of RoI
    roiSizeVoxels = np.round(RoIparameters[4] / voxelSize)
    print "[{0:.2f} s]".format(time.time() - start) + "    - Calculating RoI voxel size: ", roiSizeVoxels;
    for i in range(3):
        if(roiSizeVoxels[i]/2 == 0):
            roiSizeVoxels[i] += 1;


    roiSizeVoxels = (roiSizeVoxels - 1)/2
    limits = np.zeros((6))
    limits[0] = RoIparameters[1] - roiSizeVoxels[0]
    limits[1] = RoIparameters[1] + roiSizeVoxels[0]
    limits[2] = RoIparameters[2] - roiSizeVoxels[1]
    limits[3] = RoIparameters[2] + roiSizeVoxels[1]
    limits[4] = RoIparameters[3] - roiSizeVoxels[2]
    limits[5] = RoIparameters[3] + roiSizeVoxels[2]
















    # Load image matrix
    print "[{0:.2f} s]".format(time.time() - start) + "    - Reading image..."

    vtkReader = vtk.vtkNIFTIImageReader()
    vtkReader.SetFileName(imagePath)
    vtkReader.Update()
    voxelSize = np.array(vtkReader.GetOutput().GetSpacing());

    # Get matrix data
    image = vtkhelper.ImageData_to_array(vtkReader.GetOutput())

    print "image size ", image.shape, "voxelSize ", voxelSize

    # Calculating limits of RoI
    roiSizeVoxels = np.round(RoIparameters[4] / voxelSize)
    print "[{0:.2f} s]".format(time.time() - start) + "    - Calculating RoI voxel size: ", roiSizeVoxels;
    for i in range(3):
        if(roiSizeVoxels[i]/2 == 0):
            roiSizeVoxels[i] += 1;


    roiSizeVoxels = (roiSizeVoxels - 1)/2
    limits = np.zeros((6))
    limits[0] = RoIparameters[1] - roiSizeVoxels[0]
    limits[1] = RoIparameters[1] + roiSizeVoxels[0]
    limits[2] = RoIparameters[2] - roiSizeVoxels[1]
    limits[3] = RoIparameters[2] + roiSizeVoxels[1]
    limits[4] = RoIparameters[3] - roiSizeVoxels[2]
    limits[5] = RoIparameters[3] + roiSizeVoxels[2]

    #try:

    # Creating mask
    print "[{0:.2f} s]".format(time.time() - start) + "    - Creating mask with limits ", limits, " in image of size ", image.shape
    roi = image[limits[0]:limits[1], limits[2]:limits[3],limits[4]:limits[5]]

    # Saving the mask
    print "[{0:.2f} s]".format(time.time() - start) + "    - Saving RoI file to " + roiImageFilepath

    roi = vtkhelper.ImageData_from_array(roi)
    # Set voxel size
    roi.SetSpacing(voxelSize[0], voxelSize[1], voxelSize[2])


    vtkWriter = vtk.vtkNIFTIImageWriter()
    vtkWriter.SetFileName(roiImageFilepath)
    vtkWriter.SetInputData(roi)
    vtkWriter.Write()
    vtkWriter.Update()

    print "[{0:.2f} s]".format(time.time() - start) + "    - Finished "

    #except:
    #    print "    - [##] Something was wrong creating the mask. Check the RoI parameters."

    print "#########################################################"
