
path = "/Data/pwatkins/nrrds/in_chunks/";
outpath = "/Data/pwatkins/nrrds/out_chunks/";

// these are per file
//files = newArray("none_probICS")

// for raw inputs
//to16bit = newArray(0,1);
//indifferentials = newArray('blur20');

// for prob inputs
//to16bit = newArray(1,0);
//indifferentials = newArray('blur15');

//files = newArray("huge_x0017_y0019_z0002_probsICS", "huge_x0019_y0022_z0002_probsICS", "huge_x0022_y0023_z0001_probsICS", "huge_x0017_y0019_z0002_probsMEM", "huge_x0019_y0022_z0002_probsMEM", "huge_x0022_y0023_z0001_probsMEM", "huge_x0017_y0023_z0001_probsICS", "huge_x0022_y0018_z0001_probsICS", "huge_x0022_y0023_z0002_probsICS", "huge_x0017_y0023_z0001_probsMEM", "huge_x0022_y0018_z0001_probsMEM", "huge_x0022_y0023_z0002_probsMEM")
files = newArray("none_x0013_y0015_z0003_probsICS", "none_x0016_y0017_z0004_probsICS", "none_x0018_y0020_z0003_probsICS", "none_x0013_y0015_z0003_probsMEM", "none_x0016_y0017_z0004_probsMEM", "none_x0018_y0020_z0003_probsMEM", "none_x0013_y0020_z0003_probsICS", "none_x0018_y0015_z0003_probsICS", "none_x0018_y0020_z0004_probsICS", "none_x0013_y0020_z0003_probsMEM", "none_x0018_y0015_z0003_probsMEM", "none_x0018_y0020_z0004_probsMEM")
to16bit = newArray(1,1,1,1,1,1,1,1,1,1,1,1);
indifferentials = newArray('blur15','blur15','blur15','blur15','blur15','blur15','blur15','blur15','blur15','blur15','blur15','blur15');

// these are fixed for all files
augments = newArray('smooth', 'sharpen', 'edges', 
        'blur10', 'blur15', 'blur20', 'blur30', 'blur40', 'blur50', 'blur60', 
        'median', 'mean', 'min', 'max', 'var',
        'grad_mag', 'grad_dir', 'laplacian', 'large_hess', 'small_hess', 'hess_ori',
        'kuwahara', 'diff_blur'
        );
xyblurs = newArray("1.0","1.5","2.0","3.0","4.0","5.0","6.0");
zblurs = newArray("0.4","0.6","0.8","1.2","1.6","2.0","2.4");
ndifferentials = 6;
diffblurs = newArray('blur15','blur50');

for( i=0; i<files.length; i=i+1 ) {
	fn = path + files[i] + '.nrrd';

	k=0; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn); run("32-bit");
	run("Smooth", "stack");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=1; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn); run("32-bit");
	run("Sharpen", "stack");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=2; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn); run("32-bit");
	run("Find Edges", "stack");
	run("Nrrd ... ", "nrrd="+fno); close();
	
	k=3;
	for( j=0; j<xyblurs.length; j=j+1 ) {
		fno = outpath + files[i] + '_' + augments[k+j] + '.nrrd'; 
		open(fn); run("32-bit");
		run("Gaussian Blur 3D...", "x="+xyblurs[j]+" y="+xyblurs[j]+" z="+zblurs[j]);
		run("Nrrd ... ", "nrrd="+fno); close();
	}	

	k=22; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	//fn1 = files[i] + '_' + diffblurs[2*i+0] + '.nrrd';
	//fn2 = files[i] + '_' + diffblurs[2*i+1] + '.nrrd'; 
	fn1 = files[i] + '_' + diffblurs[0] + '.nrrd';
	fn2 = files[i] + '_' + diffblurs[1] + '.nrrd'; 
	open(outpath + fn1); open(outpath + fn2);
	imageCalculator("Subtract create 32-bit stack", fn1,fn2);
	run("Nrrd ... ", "nrrd="+fno); close(); close(); close();
	
	k=10; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn);
	run("Median 3D...", "x=2 y=2 z=1");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=12; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn);
	run("Minimum 3D...", "x=2 y=2 z=1");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=13; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn);
	run("Maximum 3D...", "x=2 y=2 z=1");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=21; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn);
	if( to16bit[i] ) 
		run("16-bit");
	run("Kuwahara Filter", "sampling=5 stack");
	run("Nrrd ... ", "nrrd="+fno); close();
	
	k=11; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn); run("32-bit");
	run("Mean 3D...", "x=2 y=2 z=1");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=14; fno = outpath + files[i] + '_' + augments[k] + '.nrrd'; 
	open(fn); run("32-bit");
	run("Variance 3D...", "x=2 y=2 z=1");
	run("Nrrd ... ", "nrrd="+fno); close();

	k=15;
	for( j=0; j<ndifferentials; j=j+1 ) {
		fno = outpath + files[i] + '_' + augments[k+j] + '.nrrd'; 
		fn1 = files[i] + '_' + indifferentials[i] + '.nrrd';
		open(outpath + fn1);
		waitForUser( "Pause","Select\n" + augments[k+j]);
		run("Differentials");
		run("Nrrd ... ", "nrrd="+fno); close();
	}
	
}
