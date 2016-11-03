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

function [krk_output, meta, commentsString] = KLEE_readKNOSSOS_v4(krk_fname)
    % matches .nml format on Jan 21st 2009
    
    if nargin<1
        krk_fname = '';
        [filename,pathname] = uigetfile;
        if filename~=0        
            krk_fname = fullfile(pathname,filename);
        end
    end
    
    if ~isempty(krk_fname)
        fid = fopen(krk_fname,'r');
        krk_contents = fscanf(fid,'%c');
        fclose(fid);
        
        % load PARAMETERS (returned in meta), modified by pwatkins
        krk_parameters = regexp(krk_contents,'<parameters>(.*?)</parameters>','tokens');
        params = regexp(krk_parameters{1}{1},'<(\S+?)\s+(.*?)\/>','tokens');
        for krk_pc=1:length(params)
            subparams = regexp(params{krk_pc}{2},'(\S*?)=\"(.*?)\"','tokens');
            for krk_sub=1:length(subparams)
              str = '';
              if length(subparams{krk_sub}{2}) < 64
                str = str2num(subparams{krk_sub}{2});
              end
              % some knossos genius added '.' to the parameter names in the new xml format
              subparams{krk_sub}{1} = strrep(subparams{krk_sub}{1}, '.', '_');
              if isempty(str)
                meta.(params{krk_pc}{1}).(subparams{krk_sub}{1}) = subparams{krk_sub}{2};
              else
                meta.(params{krk_pc}{1}).(subparams{krk_sub}{1}) = str;
              end
            end
        end
        
        krk_things = regexp(krk_contents,'<thing id.*?</thing>','match');

        % load COMMENTS
        commentsString = regexp(krk_contents,'<comments>.*</comments>','match');
        
        % load THINGS
        
        for krk_tc=1:size(krk_things,2)
            %krk_output{krk_tc} = struct('nodes',[],'edges',[]);
            temp = regexp(krk_things{krk_tc},'<thing id.*?>','match');
            temp2 = regexp(temp{:},'"\.*"?','split');
            if length(temp2) > 11
              krk_output{krk_tc}.comment = temp2{12};
            end
            krk_theseNodes = regexp(krk_things{krk_tc},'<node .*?/>','match');
            krk_output{krk_tc}.nodes = repmat(0,[size(krk_theseNodes,2),5]);
            for krk_nc=1:size(krk_theseNodes,2)
                krk_thisNode = regexp(krk_theseNodes{krk_nc},'\".+?\"','match');
                krk_output{krk_tc}.nodes(krk_nc,:) = [str2num(krk_thisNode{1}(2:end-1)),...
                    str2num(krk_thisNode{2}(2:end-1)),...
                    str2num(krk_thisNode{3}(2:end-1)),...
                    str2num(krk_thisNode{4}(2:end-1)),...
                    str2num(krk_thisNode{5}(2:end-1))];
            end
            krk_nodeIDconversion = [];
            for krk_nc=1:size(krk_theseNodes,2)
                krk_nodeIDconversion(krk_output{krk_tc}.nodes(krk_nc,1)) = krk_nc;
            end
            krk_theseEdges = regexp(krk_things{krk_tc},'<edge .*?/>','match');
            krk_output{krk_tc}.edges = repmat(0,[size(krk_theseEdges,2),2]);
            for krk_nc=1:size(krk_theseEdges,2)
                krk_thisEdge = regexp(krk_theseEdges{krk_nc},'\".+?\"','match');
                krk_output{krk_tc}.edges(krk_nc,:) = [str2num(krk_thisEdge{1}(2:end-1)),...
                    str2num(krk_thisEdge{2}(2:end-1))];
            end
            
            if size(krk_output{krk_tc}.edges,1)>1 
            krk_output{krk_tc}.edges = krk_nodeIDconversion(krk_output{krk_tc}.edges);
            else
                krk_output{krk_tc}.edges = [];
            end
%             krk_output{krk_tc}.edges = krk_output{krk_tc}.edges;
            krk_output{krk_tc}.nodes = krk_output{krk_tc}.nodes(:,[3:5,2,1]);
            
            krk_thingID = regexp(krk_things{krk_tc},'<thing id.{6}','match');
            krk_thingID = regexp(krk_thingID{1},'[0123456789]*','match');
            krk_output{krk_tc}.thingID = str2num(krk_thingID{1});
            
        end
    end













end
