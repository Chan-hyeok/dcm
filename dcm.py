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

__version__ = '2.0.0'

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
			self.rows , self.cols = self.shape
			self.height, self.width = self.shape
			self.channel, self.batch = arg.channel, arg.batch
			self.kind = arg.kind
			self.color = arg.color
			self.dim = matrix

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
			if name[-4:].lower() == '.img':
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

	def __getitem__(self, index):
		new = DCM(self)
		new.data = self.data[index]
		new.shape = new.data.shape

		if self.dim == matrix and isinstance(index, int):
			new.dim = row_vec
			new.rows, new.height = 1, 1

		elif (self.dim == row_vec or self.dim == col_vec) and isinstance(index, int):
			new.dim = scalar
			new.rows, new.cols, new.height, new.width = 1,1,1,1

		elif isinstance(index, slice):
			if self.dim == row_vec:
				new.cols, new.width = self.shape[0], self.shape[0]
			else:
				new.rows, new.height = self.shape[0], self.shape[0]

		elif isinstance(index, tuple):
			if len(index) == 1 and isinstance(index[0], int):
				return self[index[0]]
			elif isinstance(index[0], int) and isinstance(index[1], int):
				new.dim = scalar
				new.rows, new.cols, new.width, new.height = 1,1,1,1
			elif isinstance(index[0], int) and isinstance(index[1], slice):
				new.dim = row_vec
				new.rows, new.height = 1
				new.cols, new.width = self.shape[0], self.shape[0]
			elif isinstance(index[0], slice) and isinstance(index[1], int):
				new.dim = col_vec
				new.rows, new.height = self.shape[0], self.shape[1]
				new.cols, self.width = 1
			elif isinstance(index[0], slice) and isinstance(index[1], slice):
				new.rows, new.cols = self.shape
				new.width, new.height = self.shape
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
				return np.repeat(self.data_with_channel(framework), n, axis=0)

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

def read(directory, dtype=ufloat32):
	return DCM(directory, dtype)

def read_directory(directory):
	lst = []
	paths = [p for p in Path(directory).glob("**/*") if p.suffix.lower() in [".dcm"]]
	for path in paths:
		lst.append(DCM(str(path)))
	return lst
