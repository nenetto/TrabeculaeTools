__author__ = 'Eugenio Marinetto'

import csv
import numpy as np
import vtk
import math
import os
import time
import sys
import seaborn as sns
import pandas as pd
import SimpleITK as sitk

from datetime import datetime
from os import listdir
from os.path import isfile, join, exists
from numpy import genfromtxt
from scipy import ndimage
from scipy.spatial.distance import dice, jaccard, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule

from PythonTools.helpers import vtk as vtkhelper
from PythonTools import registration, transforms, transformations, io

from pyimagej.pyimagej import ImageJ, MacroImageJ

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
# New functions
###############################################################################################

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


    RoIStructure = pd.DataFrame(columns = [ 'File', \
                                                'RoI Number', \
                                                'RoI Size mm', \
                                                'Center x mm', \
                                                'Center y mm', \
                                                'Center z mm'],
                                    index = range(nRoIs))


    for i in range(nRoIs):
        RoIStructure.iloc[i] = pd.Series({  'File' : MaskFilePath,\
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
            TotalDataFrame = pd.read_csv(csvFile,index_col=0)
            first = False
        else:
            newDataFrame = pd.read_csv(csvFile,index_col=0)
            TotalDataFrame = TotalDataFrame.append(newDataFrame, ignore_index=True)


    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Saving to File "

    TotalDataFrame.to_csv(PathToFileComplete)

    return TotalDataFrame

def CreateRoIsfromRoIFile(PathToRoIfile, ImageTargetFilePath, ImageSourceFilePath, TransformationFile, WorkingDir, PrintDebug = True):
    '''
        Read each RoI definition and create RoI files for Target and Source Images

        Keyword arguments:
        PathToRoIfile               -- csv file with RoIs definition
        ImageTargetFilePath         -- Target Image
        ImageSourceFilePath         -- Source Image (RoIs will be wrapped to Target)
        TransformationFile          -- Transformation that match Source to Target
        WorkingDir                  -- Directory where save RoIImages
        PringDebug          [True]  -- True if printing messages are wanted
    '''
    # Declare starting time for process
    startTime = time.time()

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Reading RoI Structure"

    RoIStructure = pd.read_csv(PathToRoIfile,index_col=0)
    NumberOfRoIs = len(RoIStructure.index)

    if PrintDebug:
        print "[{0:.2f} s]".format(time.time() - startTime) + "    - Number of RoIs {0:d}".format(NumberOfRoIs)

    percentage = 0
    for i in range(NumberOfRoIs):
        percentage = 100.0 * float(i) / (NumberOfRoIs-1)
        RoIDefinition = RoIStructure.iloc[i]

        RoIFilePathTarget = WorkingDir + '\\' + ImageTargetFilePath.split('\\')[-1][:-4] + '_{1:.2f}mm_RoI{0:d}.nii'.format(RoIDefinition['RoI Number'],RoIDefinition['RoI Size mm'])

        CreateRoI(  ImageFilePath = ImageTargetFilePath,\
                    RoIDefinition = RoIDefinition,\
                    RoIFilePath = RoIFilePathTarget, \
                    PrintDebug = False)

        RoIFilePathSource = WorkingDir + '\\' + ImageSourceFilePath.split('\\')[-1][:-4] + '_{1:.2f}mm_RoI{0:d}.nii'.format(RoIDefinition['RoI Number'],RoIDefinition['RoI Size mm'])

        CreateRoITransformed(ImageFilePath = ImageSourceFilePath,\
                             RoIDefinition = RoIDefinition,\
                             TransformationFile = TransformationFile,\
                             ReferenceRoIImageFilePath = RoIFilePathTarget,\
                             RoIFilePath = RoIFilePathSource,\
                             PrintDebug = False)
        PrintPercentage(percentage, preMessage = 'Creating RoIs ...')

    print "Finish!, total time:  {0:.2f}".format(time.time() - startTime)

def PrintPercentage(percentage, preMessage = ''):
    timing = datetime.now().strftime('%H:%M:%S')
    timing = "[" + timing + "]"

    nlines = (np.round(20 * percentage/100.0)).astype(int)
    newString = preMessage + timing + "--[" + '|'*nlines + ' '*(20-nlines) + "]"
    if(percentage == 100.0):
        newString = newString + ' Finished!'

    sys.stdout.write("\r%s" % newString)
    sys.stdout.flush()

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

    segmentTrabecula = macroImageJ(imagejPath = imagejPath,\
                               macroPath = macroPath,\
                               xmlDefinition = xmlDefinition,\
                               defaultTimeout = defaultTimeout)

    segmentTrabecula.runMacro(  SMOOTH_Sigma = SMOOTH_Sigma,\
                                TH_Erosion = TH_Erosion,\
                                TH_Dilation = TH_Dilation,\
                                inputImage = PathToRoIfile,\
                                outputImage = PathToSegmentedRoIfile)

    #print "Finish!, total time:  {0:.2f}".format(time.time() - startTime)

    if isfile(PathToSegmentedRoIfile):
        ResultStruct = pd.DataFrame(columns = [ 'Origin RoI',\
                                                'Segmented File',\
                                                'Segmentation Algorithm',\
                                                'Smooth Sigma',\
                                                'Number of Erosions',\
                                                'Number of Dilations',\
                                                ], index = range(1))


        ResultStruct.iloc[0]  = pd.Series({ 'Origin RoI' : PathToRoIfile,\
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
    onlyfiles = [ f for f in listdir(resultsPathData) if (isfile(join(resultsPathData,f)) & (ImageName in f) ) ]

    for f in onlyfiles:
        df = pd.read_csv(join(resultsPathData,f),index_col=0)

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
    dataMetrics = pd.DataFrame(columns = [\
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
    dataMetrics.iloc[0] = pd.Series({\
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
    dataMetricsParametersFixed

    # Join Together
    ResultingData = pd.concat([dataSegmentation, dataMetricsParametersFixed, dataMetrics], axis=1, join_axes=[dataMetrics.index])

    return ResultingData









    print "Finish!, total time:  {0:.2f}".format(time.time() - startTime)

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
    differenceDF = pd.DataFrame(columns = [\
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
    differenceDF.iloc[0] = pd.Series({\
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



    ResultingData = pd.concat([resultsTested, differenceDF], axis=1, join_axes=[resultsTested.index])

    resultsGoldStandard['Image origin'] = 'uCT'
    resultsGoldStandard['RoI Size mm'] = RoISize
    resultsGoldStandard['RoI Number'] = RoINumber
    resultsGoldStandard['Center x mm'] = RoIX
    resultsGoldStandard['Center y mm'] = RoIY
    resultsGoldStandard['Center z mm'] = RoIZ
    resultsGoldStandard['Distance to zero'] = np.sqrt(RoIX*RoIX + RoIY*RoIY + RoIZ*RoIZ)

    ResultingData = ResultingData.append(resultsGoldStandard)

    return ResultingData

def DownsampleImage(ImagePath, ResultPath, DownsamplingFactor, PrintDebug = True):

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

def launchCMD(cmdString):

    os.system("start /wait cmd /c " + cmdString)

def CreateRoIfileStudy(MaskFilePath, RoIFolder, RoISizeVector, NRandomRoIs, RandomSeed = 0, PrintDebug = False):

    np.random.seed(RandomSeed)

    if not os.path.exists(RoIFolder):
        os.makedirs(RoIFolder)

    N = len(RoISizeVector)
    percentageIt = 0.0

    PrintPercentage(0.0, preMessage = 'Creating RoI File: ')
    for SizeRoImm in RoISizeVector:
        ErodedMaskFilePath = RoIFolder + r'\ErodedBoneMask_{0:2.2f}.nii'.format(SizeRoImm)
        RoIcsvFilePath = RoIFolder + r'\RoIFile_{0:2.2f}.csv'.format(SizeRoImm)
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


    PathToFileComplete = RoIFolder + r'\RoIFileComplete.csv'

    PathToFileList = []
    for f in listdir(RoIFolder):
        if 'RoIFile_' in f:
            PathToFileList.append(RoIFolder + '\\' + f)

    RoIStructure = JoinCSVFiles(PathToFileList,PathToFileComplete, PrintDebug = False)
    PrintPercentage(100.0, preMessage = 'Creating RoI File: ')

def SetCenterForSegmentation(SegmentedRoIFilePath , RoIFilePath):

    RoIImage = sitk.Cast(sitk.ReadImage(RoIFilePath), sitk.sitkFloat32)

    voxelSize = np.array(RoIImage.GetSpacing())
    imageSize = np.array(RoIImage.GetSize())
    center =  np.array(RoIImage.GetOrigin())

    RoIImage = None
    RoIImage = sitk.Cast(sitk.ReadImage(SegmentedRoIFilePath), sitk.sitkFloat32)

    downSampleFactor = DownsamplingFactor
    itkResampler = sitk.ResampleImageFilter()
    itkResampler.SetInterpolator(sitk.sitkBSpline)
    itkResampler.SetDefaultPixelValue(0)
    itkResampler.SetTransform(sitk.Transform())
    itkResampler.SetOutputSpacing(voxelSize)
    itkResampler.SetSize(np.round(imageSize).astype(int))
    itkResampler.SetOutputOrigin(center)
    RoIImage = itkResampler.Execute(RoIImage)

    sitk.WriteImage(RoIImage, SegmentedRoIFilePath)

    RoIImage = None

def SegmentTrabeculaeBoneJData( imagejPath, macroPath, xmlDefinition, PathToRoIfile, PathToSegmentedRoIfile, defaultTimeout = 120, SMOOTH_Sigma = 0.03, TH_Erosion = 0, TH_Dilation = 0):
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

    #print "Finish!, total time:  {0:.2f}".format(time.time() - startTime)

    if isfile(PathToSegmentedRoIfile):
        ResultStruct = pd.DataFrame(columns = [ 'Origin RoI',\
                                                'Segmented File',\
                                                'Segmentation Algorithm',\
                                                'Smooth Sigma',\
                                                'Number of Erosions',\
                                                'Number of Dilations',\
                                                ], index = range(1))


        ResultStruct.iloc[0]  = pd.Series({ 'Origin RoI' : PathToRoIfile,\
                                            'Segmented File' : PathToSegmentedRoIfile,\
                                            'Segmentation Algorithm' : 'BoneJ',\
                                            'Smooth Sigma' : SMOOTH_Sigma,\
                                            'Number of Erosions' : TH_Erosion,\
                                            'Number of Dilations' : TH_Dilation})

        return ResultStruct
    else:
        PrintPercentage(90.0, preMessage = 'Error: Image did not produce results')
        return None
