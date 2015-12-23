# The MIT License (MIT)
# 
# Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from data import *
from parseEMdata import EMDataParser
import os

class EMDataProvider(LabeledDataProvider):
    # xxx - this might not be the cleanest way to do this? only ever allow one instance of EMDataParser per process
    data_parser = None
    write_features = False      # any time that emdata is used in combination with feature writing mode in convnet
    # this really should be an enumerated type indicating which feature writing mode, available options:
    #   prob    writes out the convnet output probabilities
    #   data    for initializing data pre-processing based on some pre-processing stages in convnet
    write_features_type = ''
    first_batch = 0
    
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        if EMDataProvider.data_parser == None:
            assert os.path.isfile(data_dir) # this needs to be the full path / file name of EMDataParser config file
            
            # if the convnet is writing features and an em feature path is provided then also have parser write outputs
            EMDataProvider.write_features = dp_params['convnet'].op.get_value('write_features')
            write_outputs = False; append_features = False  # modes exposed in init by the EMDataParser class 
            if EMDataProvider.write_features:
                if dp_params['em_feature_path']:
                    # this command line flag along with write_features enables writing output probabilities
                    EMDataProvider.write_features_type = 'prob'
                    # if the em_feature_path is an hdf file name, then this is a single whole-dataset hdf5 file
                    fn, ext = os.path.splitext(dp_params['em_feature_path']); ext = ext.lower()
                    append_features = (ext == '.h5' or ext == '.hdf5')
                    # if not appending features, then just do normal write outputs
                    write_outputs = not append_features
                else:
                    # if em_feature_path is not specified, then this mode is for initializing data pre-processing
                    EMDataProvider.write_features_type = 'data'
                    assert( dp_params['convnet'].op.get_value('numpy_dump') )
                
            # instantiate the parser, override some attributes and then initialize 
            EMDataProvider.data_parser = EMDataParser(data_dir, write_outputs, dp_params['init_load_path'], 
                dp_params['save_name'], append_features)
            # if writing any features, override the outpath and force no label lookup
            if EMDataProvider.write_features: 
                EMDataProvider.data_parser.outpath = dp_params['em_feature_path']
                EMDataProvider.data_parser.no_label_lookup = True
            EMDataProvider.data_parser.initBatches()
        self.batch_meta = EMDataProvider.data_parser.batch_meta
        self.batches_generated = 0
        
    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        data, labels = EMDataProvider.data_parser.getBatch(batchnum)
        self.batches_generated += 1     # xxx - not currently using this
        return epoch, batchnum, [data, labels]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_pixels_per_case'] if idx == 0 else self.batch_meta['noutputs']

    def on_finish_featurebatch(self, feature_path, batchnum, isLastbatch):
        # do nothing if EM provider is not in feature writing mode
        if EMDataProvider.write_features_type == 'prob':
            EMDataProvider.data_parser.checkOutputCubes(feature_path, batchnum, isLastbatch)
        elif EMDataProvider.write_features_type == 'data':
            if not EMDataProvider.first_batch: EMDataProvider.first_batch = batchnum
            if isLastbatch:
                EMDataProvider.data_parser.initWhitenData(feature_path, range(EMDataProvider.first_batch, batchnum+1))

