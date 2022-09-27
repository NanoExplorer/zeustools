import numpy
from os import stat

#
# MCE Flatfile handling
#


# This is what an MCE looks like.
MCE_RC = 4
MCE_COL = 8
MCE_DWORD = 4

# This block read maximum (bytes) is to keep memory usage reasonable.
MAX_READ_SIZE = int(1e9)

class HeaderFormat:
    """
    Contains description of MCE header content and structure.
    """
    def __init__(self):
        self.offsets = {
            'status': 0,
            'frame_counter': 1,
            'row_len': 2,
            'num_rows_reported': 3,
            'data_rate': 4,
            'address0_ctr': 5,
            'header_version': 6,
            'ramp_value': 7,
            'ramp_addr': 8,
            'num_rows': 9,
            'sync_box_num': 10,
            'runfile_id':  11,
            'userfield': 12,
            }
        self.header_size = 43
        self.footer_size = 1

class BitField(object):
    """
    Describes the truncation and packing of a signal into a carrier word.
    """
    def define(self, name, start, count, scale=1., signed=True):
        self.name = name
        self.start = start
        self.count = count
        self.scale = scale
        self.signed = signed
        return self

    def extract(self, data, do_scale=True):
        """
        Extracts bit field from a numpy array of 32-bit signed integers.
        Assumes a two's complement architecture!
        """
        if self.signed:
            # Integer division preserves sign
            right = 32 - self.count
            left = right - self.start
            if left != 0:
                data = numpy.array(data).astype('int32') * 2**left
            if right != 0:
                data = numpy.array(data).astype('int32') / 2**right
        else:
            # For unsigned fields, bit operations should be used
            data = (data >> self.start) & ((1 << self.count)-1)
        if not do_scale:
            return data
        return data.astype('float') * self.scale


class DataMode(dict):
    """
    A DataMode consists of a set of BitFields describing signal
    packing into MCE data words.
    """
    def __init__(self):
        dict.__init__(self)
        self.fields = []
        self.raw = False
    def define(self, *args, **kargs):
        for a in args:
            self.fields.append(a.name)
            self[a.name] = a
        for k in kargs.keys():
            if k == 'raw':
                self.raw = True
                self.raw_info = kargs[k]
        return self


#Define the MCE data modes

MCE_data_modes = { \
    '0': DataMode().define(BitField().define('error', 0, 32)),
    '1': DataMode().define(BitField().define('fb', 0, 32, 2.**-12)),
    '2': DataMode().define(BitField().define('fb_filt', 0, 32)),
    '3': DataMode().define(BitField().define('raw', 0, 32),
                           raw={'n_cols':8, 'offsettable': False}),
    '4': DataMode().define(BitField().define('fb', 14, 18),
                           BitField().define('error', 0, 14)),
    '9': DataMode().define(BitField().define('fb_filt', 8, 24, 2.**1),
                           BitField().define('fj', 0, 8)),
    '10': DataMode().define(BitField().define('fb_filt', 7, 25, 2.**3),
                            BitField().define('fj', 0, 7)),
    '11': DataMode().define(BitField().define('row', 3, 7, signed=False),
                            BitField().define('col', 0, 3, signed=False)),
    '12': DataMode().define(BitField().define('raw', 0, 32),
                            raw={'n_cols':1, 'offsettable': True}),
}


class MCEData:
    """
    Container for MCE data (single channel) and associated header and origin information.
    """
    def __init__(self):
        self.data = {}
        self.source = None
        self.start_frame = 0
        self.n_frames = 0
        self.header = None
        self.data_is_dict = True
        self.data = []
        self.channels = []
        self.chop = []

def _rangify(start, count, n, name='items'):
    """
    Interpret start as an index into n objects; interpret count as a
    number of objects starting at start.  If start is negative,
    correct it to be relative to n.  If count is negative, adjust it
    to be relative to n.
    """
    if start < 0:
        start = n + start
    if start > n:
        print('Warning: %s requested at %i, beyond available %s.' %\
            (name, start, name))
        start = n
    if count == None:
        count = n - start
    if count < 0:
        count = n - start + count
    if start + count > n:
        print('Warning: %i %s requested, exceeding available %s.' %\
            (count, name, name))
        count = n - start
    return start, count
        
    
class SmallMCEFile:
    """
    Facilitate the loading of (single channels from) raw MCE
    flat-files.  Extraction and rescaling of data content is performed
    automatically by default.

    After instantiation with a data filename, a call to Read() will
    return the detector data as an MCEData object.

    See code for 'Reset' method for list of useful attributes.
    """
    def __init__(self, filename=None, runfile=True, basic_info=True):
        """
        Create SmallMCEFile object and load description of the data
        from runfile and header.

        :param filename: path to MCE flatfile 

        :param runfile: if True (default), filename.run is used.  If False,
            no runfile is used.  Pass a string here to override the
            runfile filename.

        :param basic_info: if True (default), basic file information is
            loaded from runfile and frame header.
        """
        # Initialize basic parameters
        self.Reset()        
        self.filename = filename      # Full path to file

        # Set runfile name
        if (filename != None) and (runfile == True):
            self.runfilename = filename+'.run'
        else:
            self.runfilename = runfile

        # If the time is right, compute content and packing
        if (filename != None) and basic_info:
            self._GetPayloadInfo()
        if (self.runfilename != False) and basic_info:
            self._ReadRunfile()
            if filename != None:
                self._GetContentInfo()

    def Reset(self):
        # Describe data sources
        self.filename = None
        self.runfilename = None

        # Describe payload frame (set through _GetPayloadInfo)
        self.n_ro = 0                 # Readout frame count
        self.size_ro = 0              # Readout data payload size, in dwords, per RC.
        self.n_rc = 0                 # Number of RC reporting
        self.rc_step = 0              # RC interleaving stride (due to MAS reordering)
        self.frame_bytes = 0          # Readout frame total size, in bytes.

        # Describe data content (set through _GetContentInfo)
        self.n_frames = 0             # Number of samples per detector
        self.n_rows = 0               # Number of rows stored by RC
        self.n_cols = 0               # Number of cols stored by RC
        self.data_mode = 0            # Data mode of RCs
        self.raw_data = False         # Data is raw mode column data
        self.divid  = 1               # Period (in frames) of CC read queries to
                                      #  RC (i.e. data_rate)
        self.freq = 0.                # Mean sampling frequency, in Hz.

        # Members for storing meta-data
        self.header = None            # Becomes dict of first frame header
        self.runfile = None           # Becomes MCERunfile for this data.

    def _rfMCEParam(self, card, param, array=False, check_sys=True):
        """
        Look up MCE 'card, param' value in runfile.
        """
        data = self.runfile.Item('HEADER', 'RB %s %s'%(card,param), type='int', \
                                 array=array)
        if data == None and check_sys:
            # On SCUBA2, some things are stored in sys only.  Blech.
            data = self.runfile.Item('HEADER', 'RB sys %s'%(param),
                                     type='int', array=array)
            if data != None:
                data = data[0]
        return data

    def _GetRCAItem(self, param):
        """
        Gets 'rc? <param>' for each RC returning data, warns if the
        setting is not consistent across acq cards, and returns the
        value from the first card.
        """
        rcs = [i+1 for i,p in enumerate(self.header['_rc_present']) if p]
        #print('mce_data.py: rcs = ', rcs)
        vals = [ self._rfMCEParam('rc%i'%r, param) for r in rcs ]
        #print('mce_data.py: vals = ', vals)
        for r,v in zip(rcs[1:], vals[1:]):
            if v == None and vals[0] != None:
                print('Warning: param \'%s\' not found on rc%i.' % \
                    (param, r))
                continue
            if vals[0] != v:
                print('Warning: param \'%s\' is not consistent accross RCs.' % \
                    (param))
                break
        return vals[0]

    def _GetContentInfo(self):
        """
        Using frame header and runfile, determines how the RC data are
        packed into the CC readout frames.

        Sets members n_cols, n_rows, divid, data_mode, n_frames.
        """
        if self.runfile == None:
            if self.runfilename == False:
                raise RuntimeError( 'Can\'t determine content params without runfile.')
            self._ReadRunfile()
        # In a pinch we could get these params from the runfile.
        if self.size_ro == 0:
            raise RuntimeError( 'Can\'t determine content params without data file.')
        # Switch on firmware revision to determine 'num_cols_reported' support
        fw_rev = self._GetRCAItem('fw_rev')
        if fw_rev >= 0x5000001:
            self.n_cols = self._GetRCAItem('num_cols_reported')
            self.n_rows = self._GetRCAItem('num_rows_reported')
        else:
            self.n_cols = MCE_COL
            self.n_rows = self._rfMCEParam('cc', 'num_rows_reported', array=False)
        self.divid = self._rfMCEParam('cc', 'data_rate', array=False)
        # Get data_mode information
        self.data_mode = self._GetRCAItem('data_mode')
        dm_data = MCE_data_modes.get('%i'%self.data_mode)
        if dm_data == None:
            dm_data = MCE_data_modes['0']

        # For 50 MHz modes, the data is entirely contiguous
        if dm_data.raw:
            self.raw_data = True
            self.n_rows = 1
            self.n_cols = dm_data.raw_info['n_cols']
            self.n_frames = int(self.n_ro * self.size_ro / self.n_cols)
            self.freq = 50.e6
            return
            
        # For rectangle modes, check RC/CC packing is reasonable
        count_rc = self.n_rows * self.n_cols
        count_cc = self.size_ro
        
        # Check 1: Warn if count_rc does not fit evenly into count_cc
        if count_cc % count_rc != 0:
            print('Warning: imperfect RC->CC frame packing (%i->%i).' % \
                (count_rc, count_cc))

        # Check 2: Warn if decimation/packing is such that samples are
        #     not evenly spaced in time.
        if count_rc != count_cc:
            if count_rc * self.divid != count_cc:
                print('Warning: bizarro uneven RC->CC frame packing.')
        
        # Determine the final data count, per channel.  Any times
        # that are not represented in all channels are lost.
        self.n_frames = int((count_cc / count_rc)) * self.n_ro

        # Store mean sampling frequency
        nr, rl, dr = [self._rfMCEParam('cc', s) for s in \
                          ['num_rows', 'row_len', 'data_rate']]
        self.freq = (50.e6 / nr / rl / dr) * (count_cc / count_rc)


    def _GetPayloadInfo(self):
        """
        Determines payload parameters using the data header and file size.

        Sets members n_ro, n_rc, size_ro, frame_bytes, rc_step.
        """
        if self.filename == None:
            raise RuntimeError( 'Can\'t determine payload params without data file.')
        if self.header == None:
            self._ReadHeader()
        # Compute frame size from header data.
        self.n_rc = self.header['_rc_present'].count(True)
        # Payload size (per-RC) is the product of two numbers:
        #  mult1 'num_rows_reported'
        #  mult2 'num_cols_reported', or 8 in pre v5 firmware
        mult1 = self.header['num_rows_reported']
        mult2 = (self.header['status'] >> 16) & 0xf
        if mult2 == 0:
            mult2 = MCE_COL
        self.rc_step = mult2
        self.size_ro = mult1*mult2
        self.frame_bytes = \
            MCE_DWORD*(self.size_ro * self.n_rc + \
                           self.header['_header_size'] + self.header['_footer_size'])
        # Now stat the file to count the readout frames
        file_size = stat(self.filename).st_size
        self.n_ro = file_size // self.frame_bytes
        if file_size % self.frame_bytes != 0:
            print('Warning: partial frame at end of file.')

    def _ReadHeader(self, offset=None):
        """
        Read the frame header at file position 'offset' (bytes),
        determine its version, and store its data in self.header.
        """
        fin = open(self.filename, "rb") # ha,ha
        if offset != None:
            fin.seek(offset)
        # It's a V6, or maybe a V7.
        format = HeaderFormat()
        head_binary = numpy.fromfile(file=fin, dtype='<i4', \
                                     count=format.header_size)
        # Lookup each offset and store
        self.header = {}
        for k in format.offsets:
            self.header[k] = head_binary[format.offsets[k]]
        # Provide some additional keys to help determine frame size
        self.header['_rc_present'] = [(self.header['status'] & (1 << 10+i))!=0 \
                                          for i in range(MCE_RC)]
        self.header['_header_size'] = format.header_size
        self.header['_footer_size'] = format.footer_size

    def _ReadRunfile(self):
        """
        Load the runfile data into self.runfile_data, using the filename in self.runfile.
        Returns None if object was initialized without runfile=False
        """
        if self.runfilename == False:
            return None
        self.runfile = MCERunfile(self.runfilename)
        return self.runfile

    def ReadRaw(self, count=None, start=0, raw_frames=False):
        """
        Load data as CC output frames.  Most users will prefer the
        Read() method, which decodes the data into detector channels.

        Returns a (frames x dets) array of integers.
        """
        if self.size_ro <= 0:
            self._GetPayloadInfo()
        # Do the count logic, warn user if something is amiss
        #print(type(start),type(count),type(self.n_ro))
        start, count = _rangify(start, count, self.n_ro, 'frames')
        # Check max frame size
        if count * self.frame_bytes > MAX_READ_SIZE:
            # Users: override this by changing the value of mce_data.MAX_READ_SIZE
            print('Warning: maximum read of %i bytes exceeded; limiting.' % \
                MAX_READ_SIZE)
            count = MAX_READ_SIZE // self.frame_bytes

        # Open, seek, read.
        f_dwords = self.frame_bytes // MCE_DWORD
        fin = open(self.filename, "rb") # ha,haz
        #print(self.frame_bytes)
        fin.seek(start*self.frame_bytes)
        a = numpy.fromfile(file=fin, dtype='<i4', count=count*f_dwords)
        n_frames = len(a) // f_dwords
        if len(a) != count*f_dwords:
            print('Warning: read problem, only %i of %i requested frames were read.'% \
                  (len(a)//f_dwords, count))
        a.shape = (n_frames, f_dwords)
        if raw_frames:
            # Return all data (i.e. including header and checksum)
            return a
        else:
            # Return the detector data only
            ho = self.header['_header_size']
            return a[:,ho:ho+self.size_ro*self.n_rc]

    def _NameChannels(self, row_col=False):
        """
        Determine MCE rows and columns of channels that are read out
        in this data file.  Return as list of (row, col) tuples.  For
        raw mode data, only a list of columns is returned.
        """
        if self.runfile == None:
            self._ReadRunfile()
        rc_p = self.header['_rc_present']
        rcs = [i for i,p in enumerate(rc_p) if p]

        # Is this raw data?  Special handling.
        dm_data = MCE_data_modes['%i'%self.data_mode]
        if dm_data.raw:
            if dm_data.raw_info['offsettable']:
                offsets = [ self._rfMCEParam('rc%i'%(rc+1), 'readout_col_index') \
                                for rc in rcs ]
            else:
                offsets = [ 0 for rc in rcs]
            return [i + o + r*MCE_COL for i in range(dm_data.raw_info['n_cols']) \
                        for r,o in zip(rcs, offsets)]

        row_index = [ self._rfMCEParam('rc%i'%(r+1), 'readout_row_index') \
                          for r in rcs ]
        col_index = [ r*MCE_COL + self._rfMCEParam('rc%i'%(r+1), 'readout_col_index') \
                          for r in rcs ]
        for i in range(len(rcs)):
            if row_index[i] == None: row_index[i] = 0
            if col_index[i] == None: col_index[i] = 0


        if row_col:
            # Use the row-indexing provided by the first RC.
            rows = [i+row_index[0] for i in range(self.n_rows)]
            # Columns can be non-contiguous
            cols = [col_index[rc]+c for rc in range(self.n_rc)
                    for c in range(self.n_cols) ]
            return (rows, cols)
        else:
            # Assemble the final list by looping in the right order
            names = []
            for row in range(self.n_rows):
                for rc in range(self.n_rc):
                    r = row + row_index[rc]
                    for col in range(self.n_cols):
                        c = rc*MCE_COL + col + col_index[rc]
                        names.append((r, c))
        return names
                    
    def _ExtractRect(self, data_in):
        """
        Given CC data frames, extract RC channel data assuming
        according to data content parameters.
        """
        # Input data should have dimensions (n_cc_frames x self.size_ro*self.n_rc)
        n_ro = data_in.shape[0]

        # Reshape data_in to (cc_frame, cc_row, cc_col) so we can work
        # with each RC's data one-by-one
        data_in.shape = (n_ro, -1, self.n_rc * self.rc_step)

        # Short-hand some critical sizes and declare output data array
        f = self.n_cols*self.n_rows          # RC frame size
        p = self.size_ro // f                 # CC/RC packing multiplier
        data = numpy.zeros((self.n_rows, self.n_rc, self.n_cols, n_ro * p))

        # The only sane way to do this is one RC at a time
        for rci in range(self.n_rc):
            # Get data from this rc, reshape to (cc_frame, cc_idx)
            x = data_in[:,:,self.rc_step*rci:self.rc_step*(rci+1)].reshape(n_ro, -1)
            # Truncate partial data and reshape to RC frames
            x = x[:,0:f*p].reshape(-1, self.n_rows, self.n_cols)
            # Transpose to (rc_row, rc_col, rc_time) and store
            data[:,rci,:,:] = x.transpose((1,2,0))
        # Just return with one space, one time index.
        return data.reshape(self.n_rc*f, -1)


    def _ExtractRaw(self, data_in, n_cols=8):
        """
        Extract 50 MHz samples from raw frame data.
        """
        # In raw data modes, the RCs always return a perfect set of contiguous data.
        n_samp = data_in.shape[0] * data_in.shape[1] / self.n_rc / n_cols
        data = numpy.zeros((n_cols*self.n_rc, n_samp), dtype='int')

        # Reshape data_in to (cc_frame, cc_row, cc_col) so we can work
        # with each RC's data one-by-one
        data_in.shape = (-1, self.size_ro/self.rc_step, self.n_rc * self.rc_step)
        for rci in range(self.n_rc):
            # Get data from this rc as 1d array.
            x = data_in[:,:,self.rc_step*rci:self.rc_step*(rci+1)].reshape(-1)
            # Truncate partial data and reshape to (rc_sample, column)
            nf = n_cols * (x.shape[0] / n_cols)
            x = x[0:nf].reshape(-1, n_cols)
            # Transpose to (column, rc_sample) and store
            data[n_cols*rci:n_cols*(rci+1),:] = x.transpose()
        return data

    def Read(self, count=None, start=0, dets=None,
             do_extract=True, do_scale=True, data_mode=None,
             field=None, fields=None, row_col=False,
             raw_frames=False, cc_indices=False,
             n_frames=None):
        """
        Read MCE data, and optionally extract the MCE signals.

        :param dets:        Pass a list of (row,col) tuples of detectors to extract (None=All)
        :param count:       Number of samples to read per channel (default=None,
                    which means all of them).  Negative numbers are taken
                    relative to the end of the file.
        :param start:       Index of first sample to read (default=0).
        :param do_extract:  If True, extract signal bit-fields using data_mode from runfile
        :param do_scale:    If True, rescale the extracted bit-fields to match a reference
                    data mode.
        :param data_mode:   Overrides data_mode from runfile, or can provide data_mode if no
                    runfile is used.
        :param field:       A single field to extract.  The output data will contain an array
                    containing the extracted field.  (If None, the default field is used.)
        :param fields:      A list of fields of interest to extract, or 'all' to get all fields.
                    This overrides the value of field, and the output data will contain
                    a dictionary with the extracted field data.
        :param row_col:     If True, detector data is returned as a 3-D array with indices (row,
                    column, frame).
        :param raw_frames:  If True, return a 2d array containing raw data (including header
                    and checksum), with indices (frame, index_in_frame).
        :param cc_indices:  If True, count and start are interpreted as readout frame indices and
                    not sample indices.  Default is False.
        """
        if n_frames != None:
            print('Warning: Use of n_frames in Read() is deprecated, please use '\
                'the "count=" argument.')
            count = n_frames
        # When raw_frames is passed, count and start are passed directly to ReadRaw.
        if raw_frames:
            return self.ReadRaw(count=count, start=start, raw_frames=True)

        # We can only do this if we have a runfile
        if self.n_frames == 0:
            self._GetContentInfo()

        # Allow data_mode override
        if data_mode != None:
            self.data_mode = data_mode

        if cc_indices:
            start *= pack_factor
            if count != None:
                count *= pack_factor

        # Decode start and count arguments
        start, count = _rangify(start, count, self.n_frames, 'samples')

        # Convert sample indices to readout frame indices
        if self.raw_data:
            # Raw data is contiguous and uninterrupted
            cc_start = start * self.n_cols // self.size_ro
            cc_count = ((count+start)*self.n_cols + self.size_ro-1) // \
                self.size_ro - cc_start
        else:
            # For packed data, trim excess frame words
            pack_factor = self.size_ro // (self.n_rows * self.n_cols)
            cc_start = start // pack_factor
            cc_count = (count + start + pack_factor-1) // pack_factor - cc_start

        # Get detector data as (n_ro x (size_ro*n_rc)) array
        r_data_in = self.ReadRaw(count=cc_count, start=cc_start, raw_frames = True)
        # split data so r_data_in contains headers but data_in does not
        ho = self.header['_header_size']
        data_in = r_data_in[:,ho:ho+self.size_ro*self.n_rc]

        # Check data mode for processing instructions
        dm_data = MCE_data_modes.get('%i'%self.data_mode)
        if dm_data == None:
            print('Warning: unimplemented data mode %i, treating as 0.'%self.data_mode)
            dm_data = MCE_data_modes['0']

        # Handle data packing
        if dm_data.raw:
            # Raw mode data is automatically contiguous
            data = self._ExtractRaw(data_in, dm_data.raw_info['n_cols'])
            # Trim
            offset = start - cc_start * self.size_ro
            data = data[:, offset:offset+count]
        else:
            # Normal/rectangle data may be packed incommensurately.
            data = self._ExtractRect(data_in)
            # Trim to caller's spec
            offset = start - cc_start * pack_factor
            data = data[:, offset:offset+count]

        # Create the output object
        data_out = MCEData()
        data_out.source = self.filename
        data_out.n_frames = data.shape[1]
        data_out.header = self.header

        # Populate data_out with chopper signal
        format = HeaderFormat()
        status_addr = format.offsets["status"]
        all_statuses = r_data_in[:,status_addr]
        chop = numpy.bitwise_and(all_statuses,0x0200)/512
        data_out.chop = chop
        
        # Unravel the field= vs. fields=[...] logic
        if field == None:
            field = 'default'
        force_dict = (fields != None)
        if fields == None:
            fields = [field]
        elif fields == 'all':
            fields = dm_data.fields
        for i,f in enumerate(fields):
            if f=='default':
                fields[i] = dm_data.fields[0]                
        data_out.data_is_dict = (len(fields) > 1 or force_dict)
        if data_out.data_is_dict:
            data_out.data = {}

        # Extract each field and store
        for f in fields:
            # Use BitField.extract to get each field
            new_data = dm_data[f].extract(data, do_scale=do_scale)
            if row_col:
                new_data.shape = (self.n_rows, self.n_cols*self.n_rc, -1)
            if data_out.data_is_dict:
                data_out.data[f] = new_data
            else:
                data_out.data = new_data

        data_out.channels = self._NameChannels(row_col=row_col)
        return data_out

# Let's just hope that your MCEFile is a Small One.
MCEFile = SmallMCEFile


#
# MCE Runfile handling
#

class BadRunfile(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MCERunfile:
    def __init__(self, filename=None):
        self.filename = filename
        self.data = {}
        if filename != None:
            self.Read(filename)

    def Read(self, filename):
        f = open(filename, "r")  # text
        lines = f.readlines()
        block_name = None
        block_data = {}
        self.data = {}

        for l in lines:
            key, data = runfile_break(l)
            if key == None: continue

            if key[0] == '/':
                if (block_name != key[1:]):
                    raise BadRunfile('closing tag out of place')
                if data != '':
                    raise BadRunfile('closing tag carries data')
                self.data[block_name] = block_data
                block_name = None
                block_data = {}
            elif block_name == None:
                if data == None or data == '':
                    if key in self.data:
                        raise BadRunfile('duplicate block \'%s\''%key)
                    block_name = key
                else:
                    raise BadRunfile('key outside of block!')
            else:
                block_data[key] = data
        return self.data
    
    def Item(self, block, key, array=True, type='string'):
        if block not in self.data or key not in self.data[block]:
            return None
        data = self.data[block][key]
        if type=='float':
            f = [float(s) for s in data.split()]
            if not array and len(f) <= 1: return f[0]
            return f
        if type=='int':
            f = [int(s) for s in data.split()]
            if not array and len(f) <= 1: return f[0]
            return f
        if type!='string':
            print('Unknown type "%s", returning string.' % type)
        if array:
            return data.split()
        return data

    def Item2d(self, block, key_format, array=True, type='string',
               first = 0, count = None):
        done = False
        result = []
        row = first
        while not done:
            g = self.Item(block, key_format % row, array=array, type=type)
            if g == None:
                break
            result.append(g)
            row = row + 1
            if count != None and row - first == count:
                break
        return result

    def Item2dRC(self, block, key_format, array=True, type='string',
                 first = 0, count = None, rc_count=4, rc_start=1):
        rc_data = []
        for i in range(4):
            d = self.Item2d(block, key_format%(i+1), array=array,
                            type=type, first=first, count=count)
            if d == None:
                return None
            for column in d:
                rc_data.append(column)
        return rc_data

    def __getitem__(self, key):
        return self.data[key]


def runfile_break(s):
    reform = ' '.join(s.split())
    words = reform.split('>')
    n_words = len(words)
    
    if n_words == 0 or words[0] == '':
        return None, None

    if words[0][0] == '#':
        return None, None

    if words[0][0] != '<':
        raise BadRunfile(s)
    
    key = words[0][1:]
    data = ' '.join(words[1:])

    return key, data
