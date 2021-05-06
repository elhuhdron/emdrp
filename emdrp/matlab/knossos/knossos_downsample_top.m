

skelin = '~/Downloads/annotation.xml';
skelout = '~/Downloads/annotation_out.xml';

p = struct;
p.ds_ratio = [3 3 1];
p.use_radii = true;
p.experiment = 'K0057_D31';

tic; [minnodes, rngnodes] = knossos_downsample(skelin,skelout,p); toc
