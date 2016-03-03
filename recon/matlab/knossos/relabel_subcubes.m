
function o = relabel_subcubes(p, pdata)

fprintf(1,'getting dataset info from h5 files\n');
inf = h5info(pdata.datah5);
ind = find(strcmp({inf.Datasets.Name},p.dataset_data)); % find the dataset in the info struct
o.chunksize = inf.Datasets(ind).ChunkSize;
o.datasize = inf.Datasets(ind).Dataspace(1).Size;
o.scale = h5readatt(pdata.datah5,['/' p.dataset_data],'scale')';

o.thresholds = 1:pdata.numsegs;
o.nparams = length(o.thresholds);
o.nthresholds= o.nparams;
o.Tmins = 1; o.nTmins = 1;
use_Tmins = false;

o.nsubchunks = fix(p.nchunks ./ p.szsubchunks);
o.tsubchunks = prod(o.nsubchunks);

for prm=pdata.prm_start:o.nparams
  tlabels = 0;

  % this is specific to how many parameter dimensions there are, currently just Tmin and threshold
  [thr, tmn] = ind2sub([o.nthresholds o.nTmins],prm);
  
  if o.nTmins==1 && ~use_Tmins
    fprintf(1,'\nrelabeling at thr %.8f\n', o.thresholds(thr)); t = now;
    dset = sprintf('/%s/%.8f/%s',pdata.subgroup,o.thresholds(thr),p.dataset_lbls);
  else
    fprintf(1,'\nrelabeling Tmin %d, thr %.8f\n', o.Tmins(tmn),o.thresholds(thr)); t = now;
    dset = sprintf('/%s/%d/%.8f/%s',pdata.subgroup,o.Tmins(tmn),o.thresholds(thr),p.dataset_lbls);
  end

  for ind=1:o.tsubchunks
    [x,y,z] = ind2sub(o.nsubchunks,ind); subs = [x,y,z];
    offset = zeros(1,3); sel = (subs==1); offset(sel) = p.offset(sel);
    
    loadchunk = (pdata.chunk + subs-1);
    %fprintf(1,'Load chunk %d,%d,%d with offset %d,%d,%d\n\n',loadchunk,offset);
    o.loadsize = p.szsubchunks.*o.chunksize - offset;
    o.loadcorner = loadchunk.*o.chunksize + offset;

    Vlbls = h5read(pdata.lblsh5,dset,o.loadcorner+p.matlab_base,o.loadsize);
    
    if ~isempty(pdata.nlabels_attr)
      assert(false);  % xxx - didn't implement this
      % get nlabels from attributes
      o.types_nlabels(prm,:) = h5readatt(pdata.lblsh5,dset,pdata.nlabels_attr);
      nlabels = double(sum(o.types_nlabels(prm,:))); % do not remove ECS components
      %nlabels = double(nlabels(1)); Vlbls(Vlbls > nlabels) = 0;  % remove ECS components
    else
      % get nlabels with max, no easy way to get num ICS/ECS individually
      % xxx - currently this is only for comparing with agglomeration
      nlabels = double(max(Vlbls(:))); o.types_nlabels(prm,:) = [nlabels 0];
    end
    
    Vlbls = Vlbls + tlabels; tlabels = tlabels + nlabels;
    h5write(pdata.outlbls,dset,Vlbls,o.loadcorner+p.matlab_base,o.loadsize);
  end % for subchunks

  if ~isempty(pdata.nlabels_attr)
    assert(false);
    %     % get nlabels from attributes
    %     o.types_nlabels(prm,:) = h5readatt(pdata.lblsh5,dset,'types_nlabels');
    %     nlabels = double(sum(o.types_nlabels(prm,:))); % do not remove ECS components
    %     %nlabels = double(nlabels(1)); Vlbls(Vlbls > nlabels) = 0;  % remove ECS components
  else
    % get nlabels with max, no easy way to get num ICS/ECS individually
    % xxx - currently this is only for comparing with agglomeration
    nlabels = max(Vlbls(:)); o.types_nlabels(prm,:) = [nlabels 0];
  end
  
  display(sprintf('\t\tdone in %.3f s, nlabels = %d',(now-t)*86400,tlabels));
end % for params
