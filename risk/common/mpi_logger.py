import torch
import numpy as np
from mpi4py import MPI
from .mpi_data_utils import mpi_gather_objects
from torch.utils.tensorboard import SummaryWriter


class LoggerMPI(object):
	"""  Logs data across multiple MPI processes and pushes to TensorBoard  """

	def __init__(self, output_directory):
		"""  Construct LoggerMPI object  """
		if MPI.COMM_WORLD.Get_rank() == 0:
			self.summary_writer = SummaryWriter(log_dir=output_directory)
			self.graph_logged = False

	def log_scalar(self, key, value, x, offset):
		"""  Logs a scalar y value, using MPI to determine x value  """
		xs = mpi_gather_objects(MPI.COMM_WORLD, x)
		if MPI.COMM_WORLD.Get_rank() == 0:
			offset += np.sum(xs)
			self.summary_writer.add_scalar(key, value, offset)

	def log_mean_value(self, key, value, x, offset):
		"""
		Collects data lists from all processes and adds their means to the logs.  If normalize is True
		plots the mean over all training experiences
		"""
		values = mpi_gather_objects(MPI.COMM_WORLD, value)
		values = self.flatten_list(values)
		xs = mpi_gather_objects(MPI.COMM_WORLD, x)
		if MPI.COMM_WORLD.Get_rank() == 0:
			offset += np.sum(xs)
			if len(values) > 0:
				if None not in values:
					self.summary_writer.add_scalar(key, np.mean(values), offset)

	def log_graph(self, observations, network):
		"""  Initialize TensorBoard logging of model graph """
		if MPI.COMM_WORLD.Get_rank() == 0:
			if not self.graph_logged:
				input_obs = torch.from_numpy(observations).float()
				self.summary_writer.add_graph(network, input_obs)
			self.graph_logged = True

	def flush(self):
		if MPI.COMM_WORLD.Get_rank() == 0:
			self.summary_writer.flush()

	@staticmethod
	def flatten_list(nested_list):
		return [item for sublist in nested_list for item in sublist]
