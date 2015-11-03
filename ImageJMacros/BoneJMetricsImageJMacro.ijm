// This macro analyses the the segmented Image

print("[INFO] Starting Image Metric Straction");

//######################
// Reading Parameters
//######################
print("[INFO] Read Parameters");

VF_SurfaceResampling = 6;
SMI_VoxelResampling = 6;
SMI_MeshSmoothing = 0.5;
ANISOTROPY_Radius = 64.5;
ANISOTROPY_Vectors = 50000;
ANISOTROPY_VectorSampling = 2.300;
ANISOTROPY_MinSpheres = 100;
ANISOTROPY_MaxSpheres = 2000;
ANISOTROPY_Tol = 0.001;
ELLIPSOID_SamplingIncrement = 0.435;
ELLIPSOID_Vectors = 100;
ELLIPSOID_SkeletonPoints = 50;
ELLIPSOID_Contact = 1;
ELLIPSOID_MaxIt = 100;
ELLIPSOID_MaxDrift = 1.73205;
ELLIPSOID_GaussianSigma = 2;

inputImage = "";
outputDir = "";

// Default Change flags
VF_SurfaceResampling_change = false;
SMI_VoxelResampling_change = false;
SMI_MeshSmoothing_change = false;
ANISOTROPY_Radius_change = false;
ANISOTROPY_Vectors_change = false;
ANISOTROPY_VectorSampling_change = false;
ANISOTROPY_MinSpheres_change = false;
ANISOTROPY_MaxSpheres_change = false;
ANISOTROPY_Tol_change = false;
ELLIPSOID_SamplingIncrement_change = false;
ELLIPSOID_Vectors_change = false;
ELLIPSOID_SkeletonPoints_change = false;
ELLIPSOID_Contact_change = false;
ELLIPSOID_MaxIt_change = false;
ELLIPSOID_MaxDrift_change = false;
ELLIPSOID_GaussianSigma_change = false;

inputImage_change = false;
outputDir_change = false;


// Getting Arguments
arguments = getArgument();
argumentArray = split(arguments, "-");
print("Reading Arguments...");
for (i=0; i<argumentArray.length; i++) {
	subArgumentArray = split(argumentArray[i]," ");
	if(subArgumentArray[0] == "inputImage"){
		inputImage_change = true;
		print("\nArgument type: inputImage");
		inputImage = subArgumentArray[1];
		for (j=2; j<subArgumentArray.length; j++) {
			inputImage = inputImage + " " + subArgumentArray[j];
		}
	}else if(subArgumentArray[0] == "outputDir"){
			outputDir_change = true;
			print("\nArgument type: outputDir");
			outputDir = subArgumentArray[1] + "\\";
			for (j=2; j<subArgumentArray.length; j++) {
				outputDir = outputDir + " " + subArgumentArray[j];
			}
	}else if(subArgumentArray[0] == "VF_SurfaceResampling"){
		  VF_SurfaceResampling_change = true;
			print("\nArgument type: VF_SurfaceResampling");
			VF_SurfaceResampling = parseInt(subArgumentArray[1]);
			print("\t" + VF_SurfaceResampling);
	}else if(subArgumentArray[0] == "SMI_VoxelResampling"){
		  SMI_VoxelResampling_change = true;
			print("\nArgument type: SMI_VoxelResampling");
			SMI_VoxelResampling = parseInt(subArgumentArray[1]);
			print("\t" + SMI_VoxelResampling);
	}else if(subArgumentArray[0] == "ANISOTROPY_Vectors"){
		  ANISOTROPY_Vectors_change = true;
			print("\nArgument type: ANISOTROPY_Vectors");
			ANISOTROPY_Vectors = parseInt(subArgumentArray[1]);
			print("\t" + ANISOTROPY_Vectors);
	}else if(subArgumentArray[0] == "ANISOTROPY_MinSpheres"){
		  ANISOTROPY_MinSpheres_change = true;
			print("\nArgument type: ANISOTROPY_MinSpheres");
			ANISOTROPY_MinSpheres = parseInt(subArgumentArray[1]);
			print("\t" + ANISOTROPY_MinSpheres);
	}else if(subArgumentArray[0] == "ANISOTROPY_MaxSpheres"){
		  ANISOTROPY_MaxSpheres_change = true;
			print("\nArgument type: ANISOTROPY_MaxSpheres");
			ANISOTROPY_MaxSpheres = parseInt(subArgumentArray[1]);
			print("\t" + ANISOTROPY_MaxSpheres);
	}else if(subArgumentArray[0] == "ELLIPSOID_Vectors"){
		  ELLIPSOID_Vectors_change = true;
			print("\nArgument type: ELLIPSOID_Vectors");
			ELLIPSOID_Vectors = parseInt(subArgumentArray[1]);
			print("\t" + ELLIPSOID_Vectors);
	}else if(subArgumentArray[0] == "ELLIPSOID_SkeletonPoints"){
		  ELLIPSOID_SkeletonPoints_change = true;
			print("\nArgument type: ELLIPSOID_SkeletonPoints");
			ELLIPSOID_SkeletonPoints = parseInt(subArgumentArray[1]);
			print("\t" + ELLIPSOID_SkeletonPoints);
	}else if(subArgumentArray[0] == "ELLIPSOID_Contact"){
		  ELLIPSOID_Contact_change = true;
			print("\nArgument type: ELLIPSOID_Contact");
			ELLIPSOID_Contact = parseInt(subArgumentArray[1]);
			print("\t" + ELLIPSOID_Contact);
	}else if(subArgumentArray[0] == "ELLIPSOID_MaxIt"){
		  ELLIPSOID_MaxIt_change = true;
			print("\nArgument type: ELLIPSOID_MaxIt");
			ELLIPSOID_MaxIt = parseInt(subArgumentArray[1]);
			print("\t" + ELLIPSOID_MaxIt);
	}else if(subArgumentArray[0] == "SMOOTH_Sigma"){
		SMOOTH_Sigma_change = true;
		print("\nArgument type: SMOOTH_Sigma");
		SMOOTH_Sigma = parseFloat(subArgumentArray[1]);
		print("\t" + SMOOTH_Sigma);
	}else if(subArgumentArray[0] == "TH_Range"){
		TH_Range_change = true;
		print("\nArgument type: TH_Range");
		TH_Range = parseFloat(subArgumentArray[1]);
		print("\t" + TH_Range);
	}else if(subArgumentArray[0] == "SMI_MeshSmoothing"){
		SMI_MeshSmoothing_change = true;
		print("\nArgument type: SMI_MeshSmoothing");
		SMI_MeshSmoothing = parseFloat(subArgumentArray[1]);
		print("\t" + SMI_MeshSmoothing);
	}else if(subArgumentArray[0] == "ANISOTROPY_Radius"){
		ANISOTROPY_Radius_change = true;
		print("\nArgument type: ANISOTROPY_Radius");
		ANISOTROPY_Radius = parseFloat(subArgumentArray[1]);
		print("\t" + ANISOTROPY_Radius);
	}else if(subArgumentArray[0] == "ANISOTROPY_VectorSampling"){
		ANISOTROPY_VectorSampling_change = true;
		print("\nArgument type: ANISOTROPY_VectorSampling");
		ANISOTROPY_VectorSampling = parseFloat(subArgumentArray[1]);
		print("\t" + ANISOTROPY_VectorSampling);
	}else if(subArgumentArray[0] == "ANISOTROPY_Tol"){
		ANISOTROPY_Tol_change = true;
		print("\nArgument type: ANISOTROPY_Tol");
		ANISOTROPY_Tol = parseFloat(subArgumentArray[1]);
		print("\t" + ANISOTROPY_Tol);
	}else if(subArgumentArray[0] == "ELLIPSOID_SamplingIncrement"){
		ELLIPSOID_SamplingIncrement_change = true;
		print("\nArgument type: ELLIPSOID_SamplingIncrement");
		ELLIPSOID_SamplingIncrement = parseFloat(subArgumentArray[1]);
		print("\t" + ELLIPSOID_SamplingIncrement);
	}else if(subArgumentArray[0] == "ELLIPSOID_MaxDrift"){
		ELLIPSOID_MaxDrift_change = true;
		print("\nArgument type: ELLIPSOID_MaxDrift");
		ELLIPSOID_MaxDrift = parseFloat(subArgumentArray[1]);
		print("\t" + ELLIPSOID_MaxDrift);
	}else if(subArgumentArray[0] == "ELLIPSOID_GaussianSigma"){
		ELLIPSOID_GaussianSigma_change = true;
		print("\nArgument type: ELLIPSOID_GaussianSigma");
		ELLIPSOID_GaussianSigma = parseFloat(subArgumentArray[1]);
		print("\t" + ELLIPSOID_GaussianSigma);
	}else if(subArgumentArray[0] == "help"){
		print("\nArgument type: [BoneJ Processing]\n\n");
		print("Input Parameters");
		print("__________________________________________________________________");

		print("\n\tImage related parameters:");
		print("\t\t-inputImage: Path to the segmented trabeculae");
		print("\t\t-outputDir: Folder where results will be saved");

		print("\n\t[1] Volume Fraction:");
		print("\t\t-VF_SurfaceResampling [6]: Surface Resampling");

		print("\n\t[2] Structural Model Index (SMI):");
		print("\t\t-SMI_VoxelResampling [6]: Voxel Resampling");
		print("\t\t-SMI_MeshSmoothing [0.5]: Mesh Smoothing");

		print("\n\t[3] Anisotropy:");
		print("\t\t-ANISOTROPY_Radius [64.5]: Radius for search");
		print("\t\t-ANISOTROPY_Vectors [50000]: Number of used Vectors");
		print("\t\t-ANISOTROPY_VectorSampling [2.3]: Vector Sampling");
		print("\t\t-ANISOTROPY_MinSpheres [100]: Minimum Number of spheres");
		print("\t\t-ANISOTROPY_MaxSpheres [2000]: Maximum number of spheres");
		print("\t\t-ANISOTROPY_Tol [0.001]: Tolerance");

		print("\n\t[4] Anisotropy:");
		print("\t\t-ELLIPSOID_SamplingIncrement [0.435]: Sampling Increment");
		print("\t\t-ELLIPSOID_Vectors [100]: Number of vectors");
		print("\t\t-ELLIPSOID_SkeletonPoints [50]: Skeleton Points");
		print("\t\t-ELLIPSOID_Contact [1]: Contact");
		print("\t\t-ELLIPSOID_MaxIt [100]: Maximum Number of iterations");
		print("\t\t-ELLIPSOID_MaxDrift [1.73205]: Maximum Drift");
		print("\t\t-ELLIPSOID_GaussianSigma [2]: Gaussian Filtering Sigma");

		print("__________________________________________________________________");
		print("For more information about the processing please refers to www.bonej.org");
		wait(5000);
		eval("script", "System.exit(0);");
	}else{
		print("Argument type: " + subArgumentArray[0] + " not recornized!");
	}
}

// Information of Processing
print("\nArgument type: [BoneJ Processing]\n\n");
print("Input Parameters");
print("__________________________________________________________________");
print("\n\tImage related parameters:");
if(inputImage_change){changeMark = "\t    [*]  ";}else{changeMark = "\t\t";}
print(changeMark +"inputImage: " + inputImage);
if(outputDir_change){changeMark = "\t    [*]  ";}else{changeMark = "\t\t";}
print(changeMark +"outputDir: " + outputDir);

print("\n\t[1] Volume Fraction:");
if(VF_SurfaceResampling_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"VF_SurfaceResampling [6]: " + VF_SurfaceResampling);

print("\n\t[2] Structural Model Index (SMI):");
if(SMI_VoxelResampling_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"SMI_VoxelResampling [6]: " + SMI_VoxelResampling);
if(SMI_MeshSmoothing_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"SMI_MeshSmoothing [0.5]: " + SMI_MeshSmoothing);

print("\n\t[3] Anisotropy:");
if(ANISOTROPY_Radius_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ANISOTROPY_Radius [64.5]: " + ANISOTROPY_Radius);
if(ANISOTROPY_Vectors_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ANISOTROPY_Vectors [50000]: " + ANISOTROPY_Vectors);
if(ANISOTROPY_VectorSampling_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ANISOTROPY_VectorSampling [2.3]: " + ANISOTROPY_VectorSampling);
if(ANISOTROPY_MinSpheres_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ANISOTROPY_MinSpheres [100]: " + ANISOTROPY_MinSpheres);
if(ANISOTROPY_MaxSpheres_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ANISOTROPY_MaxSpheres [2000]: " + ANISOTROPY_MaxSpheres);
if(ANISOTROPY_Tol_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ANISOTROPY_Tol [0.001]: " + ANISOTROPY_Tol);

print("\n\t[4] Anisotropy:");
if(ELLIPSOID_SamplingIncrement_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_SamplingIncrement [0.435]: " + ELLIPSOID_SamplingIncrement);
if(ELLIPSOID_Vectors_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_Vectors [100]: " + ELLIPSOID_Vectors);
if(ELLIPSOID_SkeletonPoints_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_SkeletonPoints [50]: " + ELLIPSOID_SkeletonPoints);
if(ELLIPSOID_Contact_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_Contact [1]: " + ELLIPSOID_Contact);
if(ELLIPSOID_MaxIt_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_MaxIt [100]: " + ELLIPSOID_MaxIt);
if(ELLIPSOID_MaxDrift_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_MaxDrift [1.73205]: " + ELLIPSOID_MaxDrift);
if(ELLIPSOID_GaussianSigma_change){changeMark = "\t    [*] ";}else{changeMark = "\t\t";}
print(changeMark +"ELLIPSOID_GaussianSigma [2]: " + ELLIPSOID_GaussianSigma);

print("__________________________________________________________________");
print("[*]: Default value was changed by command line");
//######################
// Folders and paths
//######################

// Extract the image format and ImageName
subArgumentArray = split(inputImage,"\\");
Original_ROI_Image_File = subArgumentArray[subArgumentArray.length - 1];
subArgumentArray = split(Original_ROI_Image_File,".");

OriginalImage_Name = subArgumentArray[0];
for (i=1; i<subArgumentArray.length - 1; i++){
	OriginalImage_Name = OriginalImage_Name + "." + subArgumentArray[i];
}
Original_ROI_Image_File_Path = inputImage;

	// Result path

File.makeDirectory(outputDir);
ResultImagesDir = outputDir + "images";
File.makeDirectory(ResultImagesDir);
ResultDataDir = outputDir + "data";
File.makeDirectory(ResultDataDir);
print("[INFO] Making dir: " + ResultImagesDir);
print("[INFO] Making dir: " + ResultDataDir);


// Init Progress

//######################
// Input Parameters
//######################

// Open the image
print("[INFO] Read Segmented Image: " + Original_ROI_Image_File_Path);
open(Original_ROI_Image_File_Path);

//######################
// Run Skeletonize 3D
//######################
print("[INFO] Skeletonize 3D processing");
run("Skeletonise 3D");
selectImage(2);
print("[INFO] Saving file: " + ResultImagesDir + "\\" + OriginalImage_Name + "_TrabeculaeSkeleton.tif");
saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_TrabeculaeSkeleton.tif");
close();


showProgress(0.25);

//######################
// Run Analyze Skeleton
//######################
//print("[INFO] Analizing Skeleton");
//run("Analyse Skeleton", "prune=none prune show");
	// Save data
//	selectWindow("Tagged skeleton");
//	print("[INFO] Saving file: " +ResultImagesDir + "\\" + OriginalImage_Name + "_TaggedSkeleton.tif");
//	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_TaggedSkeleton.tif");
//	run("Close");
//	selectWindow("Branch information");
//	print("[INFO] Saving file: " + ResultDataDir + "\\" + OriginalImage_Name + "_TrabeculaeSkeleton_BranchInfo.csv");
//	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_TrabeculaeSkeleton_BranchInfo.csv");
//  run("Close");
//	selectWindow("Results");
//	print("[INFO] Saving file: " + ResultDataDir + "\\" + OriginalImage_Name + "_TrabeculaeSkeleton_Results.csv");
//	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_TrabeculaeSkeleton_Results.csv");
//	run("Close");
	// Close All images
//	run("Close All");

showProgress(0.27);


//######################
// Run Volume Fraction
//######################
print("[INFO] Volume Fraction");
// Open the Trabeculae Image Segmentation
print("[INFO] Open file: " + Original_ROI_Image_File_Path);
open(Original_ROI_Image_File_Path);

//VF_SurfaceResampling = 6;
print("[INFO] Volume Fraction Voxel");
run("Volume Fraction", "algorithm=Voxel surface=" + VF_SurfaceResampling + " use");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_VolumeFraction_Voxel_Results.csv");
	run("Close");
showProgress(0.28);
print("[INFO] Volume Fraction Surface");
run("Volume Fraction", "algorithm=Surface surface=" + VF_SurfaceResampling + " use");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_VolumeFraction_Surface_Results.csv");
	run("Close");

showProgress(0.3);

//######################
// Run SMI
//######################
selectWindow(Original_ROI_Image_File);
print("[INFO] Structure Model Index");
	//SMI_VoxelResampling = 6;
	//SMI_MeshSmoothing = 0.5;

run("Structure Model Index", "smi=[Hildebrand & Rüegsegger] voxel="+SMI_VoxelResampling+" mesh="+SMI_MeshSmoothing);
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_SMI_Results.csv");
	run("Close");


showProgress(0.34);

//######################
// Run Connectivity
//######################
selectWindow(Original_ROI_Image_File);



print("[INFO] Connectivity");
run("Connectivity");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_Connectivity_Results.csv");
	run("Close");

showProgress(0.37);
showProgress(0.4);

//######################
// Fractal Dimension
//######################
selectWindow(Original_ROI_Image_File);
print("[INFO] Fractal Dimension");
run("Fractal Dimension");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_FractalDimension_Results.csv");
	run("Close");
	selectWindow("Plot");
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_FractalDimension_fitting.tif");
	run("Close");


showProgress(0.45);

//######################
// Ellipsoid Factor
//######################
selectWindow(Original_ROI_Image_File);
print("[INFO] Ellipsoid Factor extraction");
	//ELLIPSOID_SamplingIncrement = 0.435;
	//ELLIPSOID_Vectors = 100;
	//ELLIPSOID_SkeletonPoints = 50;
	//ELLIPSOID_Contact = 1;
	//ELLIPSOID_MaxIt = 100;
	//ELLIPSOID_MaxDrift = 1.73205;
	//ELLIPSOID_GaussianSigma = 2;

run("Ellipsoid Factor", "sampling_increment="+ELLIPSOID_SamplingIncrement+" vectors="+ELLIPSOID_Vectors+" skeleton_points="+ELLIPSOID_SkeletonPoints+" contact="+ELLIPSOID_Contact+" maximum_iterations="+ELLIPSOID_MaxIt+" maximum_drift="+ELLIPSOID_MaxDrift+" ef_image ellipsoid_id_image volume_image axis_ratio_images flinn_peak_plot gaussian_sigma="+ELLIPSOID_GaussianSigma+" flinn_plot");
	// Save data
	selectWindow("Volume-" + Original_ROI_Image_File);
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_Volume.tif");
	run("Close");
	selectWindow("Mid_Long-" + Original_ROI_Image_File);
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_MidLong.tif");
	run("Close");
	selectWindow("Short_Mid-" + Original_ROI_Image_File);
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_ShortMid.tif");
	run("Close");
	selectWindow("EF-" + Original_ROI_Image_File);
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_EF.tif");
	run("Close");
	selectWindow("Max-ID-" + Original_ROI_Image_File );
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_EF.tif");
	run("Close");
	selectWindow("Flinn Diagram of Weighted-flinn-plot-" + Original_ROI_Image_File);
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_FlinnDiagram.tif");
	run("Close");
	selectWindow("FlinnPeaks_" + Original_ROI_Image_File );
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_FlinnPeaks.tif");
	run("Close");


showProgress(0.6);


//###########################
// Ellipsoid Factor Analysis
//###########################
print("[INFO] Ellipsoid Factor Analysis");
// Here, the results from Ellipsoid Factor are loaded and statistical values are extracted

open(ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_Volume.tif");
run("Set Measurements...", "area mean standard modal min shape integrated median skewness kurtosis area_fraction nan redirect=None decimal=3");
run("Measure");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_Volume_Results.csv");
	run("Close");
	run("Close");

open(ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_MidLong.tif");
run("Set Measurements...", "area mean standard modal min shape integrated median skewness kurtosis area_fraction nan redirect=None decimal=3");
run("Measure");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_Mid_Long_Results.csv");
	run("Close");
	run("Close");

open(ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_ShortMid.tif");
run("Set Measurements...", "area mean standard modal min shape integrated median skewness kurtosis area_fraction nan redirect=None decimal=3");
run("Measure");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_ShortMid_Results.csv");
	run("Close");
	run("Close");

open(ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_EF.tif");
run("Set Measurements...", "area mean standard modal min shape integrated median skewness kurtosis area_fraction nan redirect=None decimal=3");
run("Measure");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_EF_Results.csv");
	run("Close");
	run("Close");

open(ResultImagesDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_EF.tif");
run("Set Measurements...", "area mean standard modal min shape integrated median skewness kurtosis area_fraction nan redirect=None decimal=3");
run("Measure");
	// Save data
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_EllipsoidFactor_Max-ID_Results.csv");
	run("Close");
	run("Close");
showProgress(0.7);

//###########################
// Thickness
//###########################
selectWindow(Original_ROI_Image_File);
print("[INFO] Thickness");
run("Thickness", "thickness spacing graphic mask");
	// Save data
	selectWindow(OriginalImage_Name + "_Tb.Th");
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_Thickness_TbTh.tif");
	run("Close");
	selectWindow(OriginalImage_Name + "_Tb.Sp");
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_Thickness_TbSp.tif");
	run("Close");
	selectWindow("Results");
	saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_Thickness_Results.csv");
	run("Close");
run("Close All");

//######################
// Run Anisotropy
//######################
open(Original_ROI_Image_File_Path);
print("[INFO] Anisotropy");
	//ANISOTROPY_Radius = 64.5;
	//ANISOTROPY_Vectors = 50000;
	//ANISOTROPY_VectorSampling = 2.300;
	//ANISOTROPY_MinSpheres = 100;
	//ANISOTROPY_MaxSpheres = 2000;
	//ANISOTROPY_Tol = 0.0005;

run("Anisotropy", "auto radius="+ANISOTROPY_Radius+" vectors="+ANISOTROPY_Vectors+" vector_sampling="+ANISOTROPY_VectorSampling+" min_spheres="+ANISOTROPY_MinSpheres+" max_spheres="+ANISOTROPY_MaxSpheres+" tolerance="+ANISOTROPY_Tol+" show_plot record_eigens");
    // Save data
	selectWindow("Results");
    saveAs("Text", ResultDataDir + "\\" + OriginalImage_Name + "_Anisotropy_Results.csv");
	run("Close");
	selectWindow("Anisotropy of " + Original_ROI_Image_File);
	saveAs("Tiff", ResultImagesDir + "\\" + OriginalImage_Name + "_Anisotropy_minimization.tif");
	run("Close");

print("[INFO] Anisotropy finished");

showProgress(1);

selectWindow("Log");
saveAs("Text", outputDir + "\\" + "LOG.txt");
eval("script", "System.exit(0);");
