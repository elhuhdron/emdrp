% The MIT License (MIT)
% 
% Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% modified version of KnossosM_exportNML_v342_forGapJunctionAnalysis from kb
function knossos_write_nml(nmlout,skels,pars,comments,node_meta)

% default_dir = get_userdata(gcf,'default_dir');

if ~exist('node_meta','var') || isempty(node_meta)
  node_meta.color_r = -1;
  node_meta.color_g = -1;
  node_meta.color_b = -1;
  node_meta.rad = 1.5;
  node_meta.inVp = 0;
  node_meta.inMag = 1;
  node_meta.time = fix(now*24); % ???
end

branches = [];

fid = fopen(nmlout,'w');

% % pars = get_userdata(gcf,'pars');
% % skels = get_userdata(gcf,'skels');
% % comments = get_userdata(gcf,'comments');
% % branches = get_userdata(gcf,'branches');
% %
% % listbox_h = findobj('Tag','listbox_h');
% % theseskels = get(listbox_h,'Value')
% %
% [filename, pathname] = uiputfile({'*.nml'},...
%     'Save NML file as',default_dir);
%
% % skelfile = 'C:\k0725_skels\k0725_allGCs.maybeDS1+SACs.057.nml'
% fid = fopen(fullfile(pathname,filename),'w');
%
% fprintf(fid,'<?xml version="1.0"?>\n');
% fprintf(fid,'<things>\n');
% fprintf(fid,'  <parameters>\n');
% fprintf(fid,'    <experiment name="%s"/>\n',pars.thisexptname);
% fprintf(fid,'    <lastsavedin version="3.4.2"/>\n');
% fprintf(fid,'    <createdin version="3.4.2"/>\n');
% fprintf(fid,'    <scale x="%0.6f" y="%0.6f" z="%0.6f"/>\n',pars.thisxscale,pars.thisyscale,pars.thiszscale);
% fprintf(fid,'    <offset x="0" y="0" z="0"/>\n');
% class(pars.thistime)
% fprintf(fid,'    <RadiusLocking enableCommentLocking="0" lockingRadius="100" lockToNodesWithComment="seed"/>\n');
% fprintf(fid,'    <skeletonDisplayMode displayModeBitFlags="1"/>\n');
% fprintf(fid,'    <time ms="12056" checksum="66ceb285e7ba20912f576fb6c12c8141d403a13de563a80fc6e5555e95a8f9e4"/>\n');
% fprintf(fid,'    <activeNode id="4"/>\n');
% fprintf(fid,'    <editPosition x="851" y="269" z="1"/>\n');
% fprintf(fid,'    <skeletonVPState E0="0.866025" E1="0.286788" E2="0.409576" E3="0.000000" E4="-0.500000" E5="0.496732" E6="0.709407" E7="0.000000" E8="0.000000" E9="0.819152" E10="-0.573576" E11="0.000000" E12="7168.000000" E13="7168.000000" E14="-7168.000000" E15="1.000000" translateX="0.000000" translateY="0.000000"/>\n');
% fprintf(fid,'    <vpSettingsZoom XYPlane="1.000000" XZPlane="1.000000" YZPlane="1.000000" SkelVP="0.000000"/>\n');
% fprintf(fid,'    <idleTime ms="0" checksum="df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"/>\n');
%
% % fprintf(fid,'    <time ms="%d"/>\n',pars.thistime);
% % fprintf(fid,'    <activeNode id="%d"/>\n',pars.thisactivenode);
% % fprintf(fid,'    <editPosition x="%d" y="%d" z="%d"/>\n',pars.thiseditX,pars.thiseditY,pars.thiseditZ);
% fprintf(fid,'  </parameters>\n');

fprintf(fid,'<?xml version="1.0" encoding="UTF-8"?>\n');
fprintf(fid,'<things>\n');
fprintf(fid,'  <parameters>\n');

param_fields = fields(pars);
for i=1:length(param_fields)
  fprintf(fid,'    <%s',param_fields{i});
  subparam_fields = fields(pars.(param_fields{i}));
  for j=1:length(subparam_fields)
    if ischar(pars.(param_fields{i}).(subparam_fields{j}))
      fprintf(fid,' %s="%s"',subparam_fields{j},pars.(param_fields{i}).(subparam_fields{j}));
    else
      fprintf(fid,' %s="%s"',subparam_fields{j},num2str(pars.(param_fields{i}).(subparam_fields{j})));
    end
    
  end
  fprintf(fid,'/>\n');
end

fprintf(fid,'  </parameters>\n');


for c = 1:length(skels)
  %     c = theseskels(c1);
  if(~isempty(skels{c}))
    fprintf(fid,'  <thing id="%d" color.r="%0.6f" color.g="%0.6f" color.b="%0.6f" color.a="1.000000" comment="">\n',...
      skels{c}.thingid,node_meta.color_r,node_meta.color_g,node_meta.color_b);
    fprintf(fid,'    <nodes>\n');
    for n = 1:size(skels{c}.nodes,1)
      a1 = skels{c}.nodes(n,1);
      a2 = node_meta.rad;
      a3 = skels{c}.nodes(n,2);
      a4 = skels{c}.nodes(n,3);
      a5 = skels{c}.nodes(n,4);
      a6 = node_meta.inVp;
      a7 = node_meta.inMag;
      a8 = node_meta.time;
      %       a1 = skels{c}.nodes(n,1);
      %       a2 = skels{c}.nodes(n,2);
      %       a3 = skels{c}.nodes(n,3);
      %       a4 = skels{c}.nodes(n,4);
      %       a5 = skels{c}.nodes(n,5);
      %       a6 = skels{c}.nodes(n,6);
      %       a7 = skels{c}.nodes(n,7);
      %       a8 = skels{c}.nodes(n,8);
      fprintf(fid,'      <node id="%d" radius="%0.6f" x="%d" y="%d" z="%d" inVp="%d" inMag="%d" time="%d"/>\n',a1,a2,a3,a4,a5,a6,a7,a8);
    end
    fprintf(fid,'    </nodes>\n');
    fprintf(fid,'    <edges>\n');
    for n = 1:size(skels{c}.edges,1)
      a1 = skels{c}.edges(n,1);
      a2 = skels{c}.edges(n,2);
      fprintf(fid,'      <edge source="%d" target="%d"/>\n',a1,a2);
    end
    fprintf(fid,'    </edges>\n');
    fprintf(fid,'  </thing>\n');
  end
end

%length(comments)
fprintf(fid,'  <comments>\n');
for n = 1:length(comments)
  fprintf(fid,'    <comment node="%d" content="%s"/>\n',comments{n}.node,comments{n}.content);
end
fprintf(fid,'  </comments>\n');

fprintf(fid,'  <branchpoints>\n');
for n = 1:length(branches)
  fprintf(fid,'    <branchpoint id="%d"/>\n',branches{n}.id);
end
fprintf(fid,'  </branchpoints>\n');

fprintf(fid,'</things>\n');

fclose(fid);
