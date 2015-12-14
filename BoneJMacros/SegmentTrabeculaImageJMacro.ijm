// This macro analyses the the segmented Image

//setBatchMode(true)

print("[INFO] Starting Image Metric Straction");

//######################
// Reading Parameters
//######################
print("[INFO] Read Parameters");

SMOOTH_Sigma = 0.01
TH_Tests = 11;
TH_Range = 0.20;
TH_Subvolume = 256;
TH_Erosion = 1;
TH_Dilation = 2;

inputImage = ""
outputImage = ""

// Default Change flags
SMOOTH_Sigma_change = false;
TH_Tests_change = false;
TH_Range_change = false;
TH_Subvolume_change = false;
TH_Erosion_change = false;
TH_Dilation_change = false;


inputImage_change = false;
outputImage_change = false;


// Getting Arguments
arguments = getArgument();
argumentArray = split(arguments, "-");
print("Reading Arguments...")
for (i=0; i<argumentArray.length; i++) {
	subArgumentArray = split(argumentArray[i]," ");

	if(subArgumentArray[0] == "inputImage"){
		inputImage_change = true;
		print("\nArgument type: inputImage");
		inputImage = subArgumentArray[1];
		for (j=2; j<subArgumentArray.length; j++) {
			inputImage = subArgumentArray[j] + " " + stringArg;
		}
		print("\t\"" + inputImage + "\"");
	}else if(subArgumentArray[0] == "outputImage"){
		outputImage_change = true;
		print("\nArgument type: outputImage");
		outputImage = subArgumentArray[1];
		for (j=2; j<subArgumentArray.length; j++) {
			outputImage = subArgumentArray[j] + " " + stringArg;
		}
		print("\t\"" + outputImage + "\"");
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
	}else if(subArgumentArray[0] == "TH_Tests"){
		  TH_Tests_change = true;
			print("\nArgument type: TH_Tests");
			TH_Tests = parseInt(subArgumentArray[1]);
			print("\t" + TH_Tests);
	}else if(subArgumentArray[0] == "TH_Subvolume"){
		  TH_Subvolume_change = true;
			print("\nArgument type: TH_Subvolume");
			TH_Subvolume = parseInt(subArgumentArray[1]);
			print("\t" + TH_Subvolume);
	}else if(subArgumentArray[0] == "TH_Erosion"){
		  TH_Erosion_change = true;
			print("\nArgument type: TH_Erosion");
			TH_Erosion = parseInt(subArgumentArray[1]);
			print("\t" + TH_Erosion);
	}else if(subArgumentArray[0] == "TH_Dilation"){
		  TH_Dilation_change = true;
			print("\nArgument type: TH_Dilation");
			TH_Dilation = parseInt(subArgumentArray[1]);
			print("\t" + TH_Dilation);
	}else if(subArgumentArray[0] == "help"){
		print("\nArgument type: [BoneJ Processing]\n\n");
		print("Input Parameters");
		print("__________________________________________________________________");

		print("\n\tImage related parameters:");
		print("\t\t-inputImage: Name of the image being processed");
		print("\t\t-outputImage_change: File format of the segmented image");

		print("\n\t[1] Smoothing:");
		print("\t\t-SMOOTH_Sigma [2.5]: Sigma for Gaussian Filtering");

		print("\n\t[2] Optimise Threshold:");
		print("\t\t-TH_Tests [11]: Number of test to perform");
		print("\t\t-TH_Range [0.2]: Range of Thresholds for testing");
		print("\t\t-TH_Subvolume [256]: Threshold subvolume");
		print("\t\t-TH_Erosion [1]: Number of Erosions");
		print("\t\t-TH_Dilation [2]: Number of Dilations");

		print("__________________________________________________________________");
		print("For more information about the processing please refers to www.bonej.org");
		eval("script", "System.exit(0);");


	}else if(subArgumentArray[0] == "xml"){
		print("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
		print("<imagejmacro>");
		print("    <title>Segment Trabecula</title>");
		print("    <description>This Macro uses the BoneJ Auto Thresholding for segment the trabecula in a RoI image</description>");
		print("    <category>ImageJ.BoneJ</category>");
		print("    <contributor>E. Marinetto</contributor>");
		print("");
		print("    <parameters>");
		print("        <label>SegmentTrabecula Macro</label>");
		print("        <description>");
		print("            Input Parameters");
		print("        </description>");
		print("");
		print("       <file>");
		print("           <name>Input Image</name>");
		print("           <flag>inputImage</flag>");
		print("           <description>Image to be thresholded</description>");
		print("       </file>");
		print("");
		print("       <file>");
		print("           <name>Output Image</name>");
		print("           <flag>outputImage</flag>");
		print("           <description>Trabecula segmentation</description>");
		print("       </file>");
		print("");
		print("       <double>");
		print("           <name>Smooth Sigma</name>");
		print("           <flag>SMOOTH_Sigma</flag>");
		print("           <description>Sigma for Gaussian Filtering</description>");
		print("           <default>2.5</default>");
		print("           <constraints>");
		print("               <minimum>0</minimum>");
		print("           </constraints>");
		print("       </double>");
		print("");
		print("       <integer>");
		print("           <name>Threshold Test Number</name>");
		print("           <flag>TH_Tests</flag>");
		print("           <description>Number of test to perform</description>");
		print("           <default>11</default>");
		print("           <constraints>");
		print("               <minimum>1</minimum>");
		print("           </constraints>");
		print("       </integer>");
		print("");
		print("       <double>");
		print("           <name>Threshold range</name>");
		print("           <flag>TH_Range</flag>");
		print("           <description>Range of Thresholds for testing</description>");
		print("           <default>0.2</default>");
		print("           <constraints>");
		print("               <minimum>0.1</minimum>");
		print("           </constraints>");
		print("       </double>");
		print("");
		print("       <integer>");
		print("           <name>Threshold Subvolume</name>");
		print("           <flag>TH_Subvolume</flag>");
		print("           <description>Threshold subvolume</description>");
		print("           <default>256</default>");
		print("           <constraints>");
		print("               <minimum>0</minimum>");
		print("           </constraints>");
		print("       </integer>");
		print("");
		print("       <integer>");
		print("           <name>Threshold Erosions</name>");
		print("           <flag>TH_Erosion</flag>");
		print("           <description>Number of Erosions</description>");
		print("           <default>1</default>");
		print("           <constraints>");
		print("               <minimum>0</minimum>");
		print("           </constraints>");
		print("       </integer>");
		print("");
		print("       <integer>");
		print("           <name>Threshold Dilations</name>");
		print("           <flag>TH_Dilation</flag>");
		print("           <description>Number of Dilations</description>");
		print("           <default>2</default>");
		print("           <constraints>");
		print("               <minimum>0</minimum>");
		print("           </constraints>");
		print("       </integer>");
		print("   </parameters>");
		print("</imagejmacro>");
		eval("script", "System.exit(0);");

	}else{
		print("Argument type: " + subArgumentArray[0] + " not recornized!");
	}
}


// Init Progress
showProgress(0.0);

//######################
// Input Parameters
//######################

// Open the image
run("NIfTI-Analyze", "open=" + inputImage);
showProgress(0.1);

//######################
// Smoothing
//######################
print("[INFO] Smooth processing");
run("16-bit");
//SMOOTH_sigma = 2.5
selectImage(1)
//run("Smooth (3D)", "method=Gaussian sigma=" + SMOOTH_Sigma + " use");
//selectImage(2);

showProgress(0.5);

//######################
// Run Thresholding
//######################
print("[INFO] Thresholding processing");
	//TH_tests = 11;
	//TH_range = 0.20;
	//TH_subvolume = 256;
	//TH_erosion = 1;
	//TH_dilation = 2;

run("Optimise Threshold", "apply show tests=" + TH_Tests + " range=" + TH_Range + " subvolume=" + TH_Subvolume + " erosion=" + TH_Erosion + " dilation=" + TH_Dilation + "");

print("[INFO] Run Purify avoiding negative Connectivity");
run("Purify", "labelling=Multithreaded chunk=4");

run("Purify", "labelling=Mapped chunk=4 make_copy");
selectWindow("Purified");
run("Dilate (3D)", "iso=255");
run("Erode (3D)", "iso=255");
run("NIfTI-1", "save=" + outputImage);

showProgress(1);
eval("script", "System.exit(0);");

