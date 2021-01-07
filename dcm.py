#created by Hunnam Chanhyeok Lee
import importlib
import pydicom as pd

from pathlib import Path
import numpy as np
import dct
dcmtorch = importlib.util.find_spec('torch')
if dcmtorch is not None:
	import dcmtorch
import cv2

__version__ = '2.0.1'

n_bits = 12
max_value = 2**n_bits - 1

"""default data types"""
uint8 = np.uint8
uint16 = np.int16
float16 = np.float16
ufloat32 = np.float32

"""data types for convenience"""
int16 = np.int16
float32 = np.float32
uint12 = np.int16

normal = 'normal'
diff = 'diff'

bone_white='bone_white'
bone_black='bone_black'

scalar='scalar'
row_vec='row_vec'
col_vec='col_vec'
matrix='matrix'
channels_first = 'channels_first'
channels_last = 'channels_last'
BHWC='BHWC'
BCHW='BCHW'

def __clip__(data, dtype, kind=normal):
	if dtype == uint8 and kind == normal:
		return np.array(np.clip(data, 0, 255), dtype=np.uint8)
	if dtype == uint8 and kind == normal:
		return np.array(np.clip(data, -255, 255), dtype=np.int16)
	s = 0. if kind == normal else -1.
	if dtype == uint16:
		return np.array(np.clip(data, max_value*s, max_value), dtype=np.int16)
	if dtype == float16:
		return np.array(np.clip(data, -1.+s, 1.-s), dtype=np.float16)
	if dtype == ufloat32:
		return np.array(np.clip(data, -1.*s, 1.), dtype=np.float32)


def __change_type__(data, fr, dtype, kind=normal):
	"""if from-data type and to-data-type are identical, just return the copy of data"""
	if fr == dtype:
		return data * 1

	dt = np.int16 if (kind==diff and dtype==uint8) else dtype
	delta = 1. if kind==normal else 0.

	if fr == uint8 and dtype == uint16:
		return np.array(data*(2**(n_bits-8)), dtype=np.int16)
	if fr == uint8 and dtype == float16:
		return np.array(data/128. - delta, dtype=np.float16)
	if fr == uint8 and dtype == ufloat32:
		return  np.array(data/256., dtype=np.float32)

	if fr == uint16 and dtype == uint8:
		return np.array(data / (2**(n_bits-8)), dtype=dt)
	if fr == uint16 and dtype == float16:
		return np.array(data*2./(2**n_bits) - delta, dtype=np.float16)
	if fr == uint16 and dtype == ufloat32:
		return  np.array(data*1./(2**n_bits), dtype=np.float32)

	if fr == float16 and dtype == uint8:
		return np.array((data+delta) * 128, dtype=dt)
	if fr == float16 and dtype == uint16:
		return np.array((data+delta)*(2**(n_bits-1)), dtype=np.int16)
	if fr == float16 and dtype == ufloat32:
		return (np.array(data, dtype=np.float32) + delta) / 2

	if fr == ufloat32 and dtype == uint8:
		return np.array(data*256, dtype=dt)
	if fr == ufloat32 and dtype == uint16:
		return np.array(data*(2**n_bits), dtype=np.int16)
	if fr == ufloat32 and dtype == float16:
		return np.array(data*2 - delta, dtype=np.float16)
	else:
		raise TypeError(dtype)

class DCM:
	def __init__(self, arg, dtype=ufloat32, color=None, height=2430, width=1994, min_value=0, max_value=max_value, big_endian=False):
		if isinstance(arg, str):
			self.directory = arg
			if arg[-4:].lower() == ".dcm":
				ds = pd.dcmread(arg)
				self.data = __change_type__(ds.pixel_array, uint16, dtype)
				if color:
					self.color = color
				else:
					self.color = bone_white

			elif arg[-4:].lower() == '.img':
				fa = open(arg, "rb")
				adata = fa.read(height*width*2)
				fa.close()
				if big_endian:
					data = np.frombuffer(adata, dtype=np.uint8)
					data = data.reshape((height*width, 2))
					new_data = np.empty((height*width, 2), dtype=np.uint8)
					new_data[:, 0], new_data[:, 1] = data[:, 1], data[:, 0]
					data = np.frombuffer(new_data.data, dtype=np.uint16)
				else:
					data = np.frombuffer(adata, dtype=np.uint16)
				ds = np.empty((height, width), dtype=np.uint16)
				ds[:] = np.reshape(data, (height, width))
				ds[ds<min_value] = min_value
				ds[:] = (ds - min_value) * (max_value/(max_value-min_value))
				if color:
					self.color = color
				else:
					self.color = bone_black
				self.data = __change_type__(ds, uint16, dtype)

			else:
				ds = cv2.imread(arg, cv2.IMREAD_GRAYSCALE)
				self.data =__change_type__(ds, uint8, dtype)
				if color:
					self.color = color
				else:
					self.color = bone_black

			self.shape = self.data.shape
			self.rows, self.cols = self.shape
			self.height, self.width = self.shape
			self.channel, self.batch = None, None
			self.dtype = dtype
			self.kind = normal
			self.dim = matrix
		else:
			self.directory = arg.directory
			self.dtype = arg.dtype
			self.data = arg.data
			self.shape = arg.shape
			self.dim = arg.dim
			if self.dim == scalar:
				pass
			elif self.dim == row_vec:
				self.width, self.cols = arg.width, arg.cols
			elif self.dim == col_vec:
				self.height, self.rows = arg.height, arg.rows
			elif self.dim == matrix:
				self.height, self.width = arg.shape
				self.rows, self.cols = arg.shape
			elif self.dim == channels_first or self.dim == channels_last:
				self.channel = arg.channel
				self.height, self.width = arg.height, arg.width
				self.rows, self.cols = arg.rows, arg.cols
			elif self.dim == BCHW or self.dim == BHWC:
				self.batch, self.channel = arg.batch, arg.channel
				self.height, self.width = arg.height, arg.width
				self.rows, self.cols = arg.rows, arg.cols
			self.kind = arg.kind
			self.color = arg.color

	def __copy__(self):
		return DCM(self)
	def __deep_copy__(self):
		new = DCM(self)
		new.data = self.data*1

	def __str__(self):
		return "DCM(directory="+self.directory+", dtype="+str(self.dtype)+", rows="+str(self.rows)+", cols="+str(self.cols)+", \n"+str(self.data)+")"
		
	def astype(self, dtype):
		new = DCM(self)
		new.data = __change_type__(self.data, self.dtype, dtype, self.kind)
		new.dtype = dtype
		return new

	def save_as(self, name):
		if self.kind == normal:
			if name[-4:].lower() == '.dcm':
				ds = pd.dcmread(self.directory)
				temp = ds.PixelData
				if self.color == bone_white:
					ds.PixelData = __change_type__(__clip__(self.data, self.dtype), self.dtype, uint16).tobytes()
				else:
					ds.PixelData = __change_type__(__clip__((-self).data, self.dtype), self.dtype, uint16).tobytes()
				ds.Rows, ds.Columns = len(self.data), 0 if self.rows == 0 else self.cols
				ds.save_as(name)
			elif name[-4:].lower() == '.img':
				raise NotImplementedError(name)
			else:
				cv2.imwrite(name, __change_type__(__clip__(self.data, self.dtype), self.dtype, uint8))
	
		elif self.kind == diff:
			data = self.astype(uint8).data_with_channel('channels_last', 3)
			maximum, minimum = np.max(data), np.min(data)
			m = maximum if maximum > -minimum else -minimum
			if m != 0:			
				data[:,:,2] = np.clip(data[:,:,2], 0, m)
				data[:,:,1] = np.clip(data[:,:,1]*(-1), 0, m)
				data[:,:,0] = data[:,:,0]*0
			cv2.imwrite(name, np.array(data*256./(m+1), dtype=np.uint8))
		return self

	def __sub__(self, other):
		new = DCM(self)
		if isinstance(other, DCM):
			if self.rows != other.rows:
				raise ValueError(self.rows, other.rows)
			if self.cols != other.cols:
				raise ValueError(self.cols, other.cols)
			if self.dtype != other.dtype:
				other = other.astype(self.dtype)
			if self.kind != self.kind:
				return self + (-other)
			if self.color != other.color:
				other = -other

			new.kind = diff
			if self.dtype == uint8:
				new.data = np.array(self.data, dtype=np.int16) - other.data
			else:
				new.data = self.data - other.data 
		else:
			new.data = self.data - other
		return new

	def __neg__(self):
		new = DCM(self)
		if self.dtype == uint8:
			new.data = 255 - self.data
		elif self.dtype == int16:
			new.data = max_value - self.data
		elif self.dtype == float16:
			new.data = -self.data
		elif self.dtype == ufloat32:
			new.data = 1 - self.data

		if self.color == bone_white:
			self.color = bone_black
		else:
			self.color = bone_white
		return new

	def __add__(self, other):
		new = DCM(self)
		if isinstance(other, DCM):
			if self.rows != other.rows:
				raise ValueError(self.rows, other.rows)
			if self.cols != other.cols:
				raise ValueError(self.cols, other.cols)
			if self.dtype != other.dtype:
				other = other.astype(self.dtype)
			if self.kind == other.kind:
				return self - (-other)
			if self.color != other.color:
				otehr = -other

				new.kind = normal
			if self.dtype == uint8:
				new.data = clip(self.data+other.data, uint8)
			else:
				new.data = self.data + other.data
		else:
			new.data = self.data + other
		return new

	def __radd__(self, other):
		return self + other
	def __rsub__(self, other):
		return (-self) + other
	def __rmul__(self, other):
		return self * other

	def __mul__(self, other):
		if other < 0:
			return (-self) * (-other)
		new = DCM(self)
		new.data = new.data * other
		return new

	def __truediv__(self, other):
		if other < 0:
			return (-self) * (-other)
		new = DCM(self)
		new.data = new.data / other
		return new
	
	def __len__(self):
		return self.shape[0]

	def __getitem__(self, index):
		new = DCM(self)
		new.data = self.data[index]
		new.shape = new.data.shape
		if len(new.shape) == 0:
			new.batch, new.channel = None, None
			new.height, new.width = None, None
			new.rows, new.cols = None, None
			new.dim = scalar

		elif self.dim == row_vec:
			new.width, new.cols = new.shape[0], new.shape[0]
		elif self.dim == col_vec:
			new.height, new.rows = new.shape[0], new.shape[0]
				
		elif self.dim == matrix:
			if len(new.shape) == 1:
				if isinstance(index, int) or isinstance(index[0], int):
					new.height, new.rows = None, None
					new.width, new.cols = new.shape[0], new.shape[0]
					new.dim = row_vec
				else:
					new.height, new.rows = new.shape[0], new.shape[0]
					new.width, new.cols = None, None
					new.dim = col_vec
			else:
				new.height, new.width = new.shape
				new.rows, new.cols = new.shape

		elif self.dim == channels_first:
			if len(new.shape) == 1:
				if not isinstance(index[0], int):
					raise NotImplementedError(index)
				if isinstance(index[1], int):
					new.channel = None
					new.height, new.rows = None, None
					new.width, new.cols = new.shape[0], new.shape[0]
					new.dim = row_vec
				else:
					new.channel = None
					new.height, new.rows = new.shape[0], new.shape[0]
					new.width, new.cols = None, None
					new.dim = col_vec
			elif len(new.shape) == 2:
				if not isinstance(index, int) and not isinstance(index[0], int):
					raise NotImplementedError(index)
				new.batch = None
				new.height, new.width = new.shape
				new.rows, new.cols = new.shape
				new.dim = matrix
			else:
				new.batch, new.height, new.width = new.shape
				_, new.rows, new.cols = new.shape

		elif self.dim == channels_last:
			if len(new.shape) == 1:
				if len(index) < 3 or not isinstance(index[2], int):
					raise NotImplementedError(index)
				if isinstance(index[0], int):
					new.channel = None
					new.height, new.rows = None, None
					new.width, new.cols = new.shape[0], new.shape[0]
					new.dim = row_vec
				else:
					new.channel = None
					new.height, new.rows = new.shape[0], new.shape[0]
					new.width, new.cols = None, None
					new.dim = col_vec
			elif len(new.shape) == 2:
				if len(index) < 3 or not isinstance(index[2], int):
					raise NotImplementedError(index)
				new.channel = None
				new.height, new.width = new.shape
				new.rows, new.cols = new.shape
				new.dim = matrix
			else:
				new.height, new.width, new.batch = new.shape
				new.height, new.width, _ = new.shape

		elif self.dim == BCHW:
			if len(new.shape) == 1:
				if not isinstance(index[0], int) or not isinstance(index[1], int):
					raise NotImplementedError(index)
				if isinstance(index[2], int):
					new.batch, new.channel = None, None
					new.height, new.rows = None, None
					new.width, new.cols = new.shape[0], new.shape[0]
					new.dim = row_vec
				else:
					new.batch, new.channel = None, None
					new.height, new.rows = new.shape[0], new.shape[0]
					new.width, new.cols = None, None
					new.dim = col_vec
			elif len(new.shape) == 2:
				if not isinstance(index[0], int) or not isinstance(index[1], int):
					raise NotImplementedError(index)
				new.batch, new.channel = None, None
				new.height, new.width = new.shape
				new.rows, new.cols = new.shape
				new.dim = matrix
			elif len(new.shape) == 3:
				if not isinstance(index, int) and not isinstance(index[0], int):
					raise NotImplementedError(index)
				new.batch = None
				new.channel, new.height, new.width = new.shape
				_, new.rows, new.cols = new.shape
				new.dim = channels_first
			else:
				new.batch, new.channel, new.height, new.width = new.shape
				_, _, new.rows, new.cols = new.shape
		
		elif self.dim == BHWC:
			if len(new.shape) == 1:
				if len(index) < 4 or (not isinstance(index[0], int) or not isinstance(index[3], int)):
					raise NotImplementedError(index)
				if isinstance(index[1], int):
					new.batch, new.channel = None, None
					new.height, new.rows = None, None
					new.width, new.cols = new.shape[0], new.shape[0]
					new.dim = row_vec
				else:
					new.batch, new.channel = None, None
					new.height, new.rows = new.shape[0], new.shape[0]
					new.width, new.cols = None, None
					new.dim = col_vec
			elif len(new.shape) == 2:
				if len(index) < 4 or (not isinstance(index[0], int) or not isinstance(index[3], int)):
					raise NotImplementedError(index)
				new.batch, new.channel = None, None
				new.height, new.width = new.shape
				new.rows, new.cols = new.shape
				new.dim = matrix
			elif len(new.shape) == 3:
				if not isinstance(index, int) and not isinstance(index[0], int):
					raise NotImplementedError(index)
				new.batch = None
				new.height, new.width, new_channel = new.shape
				new.rows, new.cols, _ = new.shape
				new.dim = channels_last
			else:
				new.batch, new.height, new.width, new.channel = new.shape
				_, new.rows, new.cols, _ = new.shape
	 
		return new

	def __setitem__(self, key, item):
		if isinstance(item, DCM):
			self.data[key] = item.data
		else:
			self.data[key] = item
	
	def __lt__(self, other):
		if isinstance(other, DCM):
			return self.data < other.data
		return self.data < other
	
	def __gt__(self, other):
		if isinstance(other, DCM):
			return self.data > other.data
		return self.data > other
	
	def __le__(self, other):
		if isinstance(other, DCM):
			return self.data <= other.data
		return self.data <= other
	
	def __ge__(self, other):
		if isinstance(other, DCM):
			return self.data >= other.data
		return self.data >= other

	def __eq__(self, other):
		if isinstance(other, DCM):
			return self.data == other.data
		return self.data == other

	def apply(self, func, **kwargs):
		new = DCM(self)
		new.data = func(new.data, **kwargs)
		return new

	def expand_dims(self, axis=None):
		new = DCM(self)

		if self.dim == scalar and axis==-1:
			new.dim = col_vec
			new.data = np.expand_dims(self.data, axis=0)
			new.height = 1
		elif self.dim == scalar:
			new.dim = row_vec
			new.data = np.expand_dims(self.data, axis=0)
			new.width = 1
		elif self.dim == row_vec:
			new.dim = matrix
			new.data = np.expand_dims(self.data, axis=0)
			new.height = 1
		elif self.dim == col_vec:
			new.dim = matrix
			new.data = np.expand_dims(self.data, axis=-1)
			new.width = 1
		elif self.dim == matrix and (axis==-1 or axis==2):
			new.dim = channels_last
			new.data = np.expand_dims(self.data, axis=-1)
			new.channel = 1
		elif self.dim == matrix:
			new.dim = channels_first
			new.data = np.expand_dims(self.data, axis=0)
			new.channel = 1
		elif self.dim == channels_last:
			new.dim = BHWC
			new.data = np.expand_dims(self.data, axis=0)
			new.batch = 1
		elif self.dim == channels_first:
			new.dim = BCHW
			new.data = np.expand_dims(self.data, axis=0)
			new.batch = 1
		new.shape = new.data.shape
		return new

	def data_with_channel(self, c='channels_last', n=1):
		if n == 0:
			return self.get_array()
		if c == 'channels_last' or c == 'tensorflow':
			if n == 1:
				return np.expand_dims(self.data, axis=2)
			else:
				return np.repeat(self.data_with_channel(), n, axis=2)
		if c == 'channels_first' or c == 'pytorch':
			if n == 1:
				return np.expand_dims(self.data, axis=0)
			else:
				return np.repeat(self.data_with_channel(c), n, axis=0)

	def rotate(self, n=1):
		new = DCM(self)
		if n % 4 ==  1:
			new.rows, new.cols = self.cols, self.rows
			new.height, new.width = new.rows, new.cols
			new.data = np.rot90(self.data)
		elif n % 4 == 2:
			new.data = np.rot90(self.data, 2)
		elif n % 4 == 3:
			new.rows, new.cols = self.cols, self.rows
			new.height, new.width = new.rows, new.cols
			new.data = np.rot90(self.data, 3)
		return new

	def flip(self, axis):
		new = DCM(self)
		new.data = np.flip(self.data, axis)
		return new
	
	def blur(self, n_filter=3, sigma=1.4):
		new = DCM(self)
		new.data = dcmtorch.blur(new.data*1, n_filter, sigma)
		return new

	def dct_suppression(self):
		new = DCM(self)
		new.data = dct.gogo(__change_type__(new.get_array(), self.dtype, uint16))
		new.data = __change_type__(new.data, uint16, self.dtype)
		return new

	def __set_shape__(self):
		self.shape = self.data.shape
		if self.dim == scalar:
			self.batch, self.channel, self.height, self.width = None, None, None, None
			self.rows, self.cols = None, None
		elif self.dim == row_vec:
			self.batch, self.channel, self.height, self.rows = None, None, None, None
			self.width, self.cols = self.shape[0], self.shape[0]
		elif self.dim == col_vec:
			self.batch, self.channel, self.width, self.rows = None, None, None, None
			self.height, self.rows = self.shape[0], self.shape[0]
		elif self.dim == matrix:
			slef.batch, self. channel = None, None
			self.height, self.width = self.shape
			self.rows, self.cols = self. shape
		elif self.dim == channels_first:
			self.batch = None
			self.channel, self.height, self.width = self.shape
			_, self.rows, self.cols = self.shape
		elif self.dim == channels_last:
			self.batch = None
			self.height, self.width, self.channel = self.shape
			self.rows, self.cols, _ = self.shape
		elif self.dim == BCHW:
			self.batch, self.channel, self.height, self.width = self.shape
			_, _, self.rows, self.cols = self.shape
		elif self.dim == BHWC:
			self.batch, self.height, self.width, self.channle = self.shape
			_, self.rows, self.cols, _ = self.shape

def concatenate(lst, axis=None):
	if axis is None:
		if lst[0].dim == channels_last:
			axis = -1
		else:
			axis = 0

	np_lst = []
	for img in lst:
		if img.dim != lst[0].dim:
			raise NotImplementedError(img.dim)
		if img.color != lst[0].color:
			raise NotImplementedError(img.color)

		np_lst.append(img.data)

	new = DCM(lst[0])
	new.data = np.concatenate(np_lst, axis=axis)	
	new.__set_shape__()
	return new

def stack(lst, axis=None):
	np_lst = []
	for img in lst:
		if img.dim != lst[0].dim:
			raise NotImplementedError(img.dim)
		if img.color != lst[0].color:
			raise NotImplementedError(img.color)

		np_lst.append(img.expand_dims(axis=axis).data)

	new = DCM(lst[0].expand_dims(axis=axis))
	new.data = np.concatenate(np_lst, axis=axis)	
	new.__set_shape__()
	return new

		

def read(path, **kwargs):
	return DCM(path, **kwargs)

def read_files(paths, dim=BCHW, **kwargs):
	lst = []
	for path in paths:
		img = read(path, **kwargs)
		if dim==channels_last or dim==BHWC:
			lst.append(img.expand_dims(axis=-1))
		else:
			lst.append(img.expand_dims(axis=0))
	return stack(lst, axis=0)

def read_directory(directory):
	lst = []
	paths = [p for p in Path(directory).glob("**/*") if p.suffix.lower() in [".dcm"]]
	for path in paths:
		lst.append(DCM(str(path)))
	return lst
