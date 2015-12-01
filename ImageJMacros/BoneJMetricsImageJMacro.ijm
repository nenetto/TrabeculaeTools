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

//######################
// Input Parameters
//######################

// Open the image
print("[INFO] Read Segmented Image: " + Original_ROI_Image_File_Path);
run("NIfTI-Analyze", "open=" + Original_ROI_Image_File_Path);

//######################
// Run Volume Fraction
//######################
print("[INFO] Volume Fraction");
// Open the Trabeculae Image Segmentation
print("[INFO] Open file: " + Original_ROI_Image_File_Path);
run("NIfTI-Analyze", "open=" + Original_ROI_Image_File_Path);


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
run("NIfTI-Analyze", "open=" + Original_ROI_Image_File_Path);
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
saveAs("Text", outputDir + "\\" + "BoneJMetricSImageJMacro_LOG.txt");
eval("script", "System.exit(0);");
