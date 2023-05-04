import sys, os, os.path
# blacklist = {'pip', 'conda','pip3'}
# sys.path = [path for path in sys.path if all(name not in path for name in blacklist)]
# os.system("pip3 install tqdm")
import builtins
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm



# def disable_print():
#     """禁用print函数"""
#     builtins.print = lambda *args, **kwargs: None

# def enable_print():
#     """启用print函数"""
#     builtins.print = print

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
import concurrent.futures
import random
def myfun(batch):
	return torch.cat([torch.unsqueeze(i[0], dim=0) for i in batch], dim=0), torch.tensor([i[1] for i in batch])
class MyDataLoader:
	def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.collate_fn = collate_fn

		self.indices = list(range(len(dataset)))
		self.current_idx = 0

		if shuffle:
			random.shuffle(self.indices)

	def __iter__(self):
		self.current_idx = 0
		return self

	def __next__(self):
		if self.current_idx >= len(self.indices):
			raise StopIteration

		batch_indices = self.indices[self.current_idx:self.current_idx+self.batch_size]
		batch = [self.dataset[i] for i in batch_indices]
		if self.collate_fn is not None:
			batch = self.collate_fn(batch)
		self.current_idx += self.batch_size
		return batch


input_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
truth_dir = os.path.join(input_dir, 'ref')
prediction_dir = os.path.join(input_dir, 'res')

# prediction_dir = './'
# truth_dir = './program/reference/reference_big'
sys.path.append(prediction_dir)
from inference import Solution

def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    if classes is None:
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)
        specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
    return specs

def classification_metrics(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    spe = np.mean(specificity(y_true, y_pred, classes={1}))
    return dict(kappa=kappa, f1=f1, spe=spe)

if __name__ == "__main__":
	print("begin evaluation...")
	if os.path.isfile(os.path.join(prediction_dir, 'result.csv')):
		start_time = time.time()
		p = os.path.join(truth_dir, 'test')
		res = []
		for i in ['0','1']:
			res+=[(j, int(i)) for j in os.listdir(os.path.join(p, i))]
		res = dict(res)
		pred = dict(np.array(pd.read_csv(os.path.join(prediction_dir, 'result.csv'))))
		y_true = np.array([res[i] for i in res.keys()])
		y_pred = np.array([pred[i] for i in res.keys()])
		count = 1

	else:
		solution = Solution()
		# disable_print()
		batch_size = solution.batch_size
		if solution.pretrained_name is not None:
			solution.model.load_state_dict(torch.load(os.path.join(prediction_dir, solution.pretrained_name)))
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		solution.model = solution.model.to(device)
		if torch.cuda.is_available() and solution.gpu_parallel:
			solution.model = nn.DataParallel(solution.model)
		if solution.is_train:
			train_dataset = datasets.ImageFolder(os.path.join(truth_dir, "train"), transform=solution.transform_train)
			val_dataset = datasets.ImageFolder(os.path.join(truth_dir, "val"), transform=solution.transform_test)
		test_dataset = datasets.ImageFolder(os.path.join(truth_dir, "test"), transform=solution.transform_test)
		
		if solution.is_train:
			train_loader = MyDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=myfun)
			val_loader = MyDataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=myfun)
		test_loader = MyDataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=myfun)

		if solution.is_train:
			num_epochs = solution.epoch
			for epoch in range(num_epochs):
				train_loss = 0
				y_true, y_pred = [], []
				# 训练模型
				for data, target in train_loader:
					data, target = data.to(device), target.to(device)
					output, loss = solution.train(data, target)
					train_loss += loss * data.size(0)
					pred = output.argmax(dim=1, keepdim=True)
					y_pred.append(np.array(pred.cpu().detach()).reshape(-1))
					y_true.append(np.array(target.cpu().detach()).reshape(-1))

				train_loss /= len(train_loader.dataset)
				y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
				metric_dict = classification_metrics(y_true, y_pred)
				spe = metric_dict['spe']
				kappa = metric_dict['kappa']
				f1 = metric_dict['f1']
				# enable_print()
				print('Train Epoch: {} Loss: {:.4f}, specificity: {:.4f},f1_score: {:.4f},kappa: {:.4f}'.format(epoch, train_loss, spe, f1, kappa))
				# disable_print()

				# 在测试集上测试模型
				val_loss = 0
				y_true, y_pred = [], []
				for data, target in val_loader:
					data, target = data.to(device), target.to(device)
					output, loss = solution.test(data, target)
					val_loss += loss * data.size(0)
					pred = output.argmax(dim=1, keepdim=True)
					y_pred.append(np.array(pred.cpu().detach()).reshape(-1))
					y_true.append(np.array(target.cpu().detach()).reshape(-1))

				val_loss /= len(val_loader.dataset)
				y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
				metric_dict = classification_metrics(y_true, y_pred)
				spe = metric_dict['spe']
				kappa = metric_dict['kappa']
				f1 = metric_dict['f1']
				# enable_print()
				print('val Epoch: {} Loss: {:.4f}, specificity: {:.4f},f1_score: {:.4f},kappa: {:.4f}'.format(epoch, val_loss, spe, f1, kappa))
				# disable_print()
		start_time = time.time()
		duration = 0
		if solution.is_parallel=='pool':
			# Create a dataloader
			print("begin parallel inference...")
			num_processes = solution.num_cpu
			print(f"Using {num_processes} CPUs for inference")
			y_pred = []
			y_true = []
			y_pred_tmp = []
			per = solution.per
			with Pool(num_processes) as p:
				for data, target in test_loader:
					data, target = data.to(device), target.to(device)
					y_true.append(np.array(target).reshape(-1))
					y_pred_tmp.append(data)
					# print("1111")
					if len(y_pred_tmp) >= per:
						mystarttime = time.time()
						y_pred_tmp = p.map(solution.inference, y_pred_tmp)
						print(y_pred_tmp)
						duration += time.time() - mystarttime
						y_pred = y_pred + y_pred_tmp
						y_pred_tmp = []
						# print("2222")
				
				if len(y_pred_tmp) > 0:
					mystarttime = time.time()
					y_pred_tmp = p.map(solution.inference, y_pred_tmp)
					duration += time.time() - mystarttime
					y_pred = y_pred + y_pred_tmp
					y_pred_tmp.clear()

			print(f"Inference parallel took {duration:.4f} seconds")
		# elif solution.is_parallel=='concurrent':
		# 	# Create a dataloader
		# 	print("begin parallel inference...")
		# 	num_processes = solution.num_cpu
		# 	print(f"Using {num_processes} CPUs for inference")
		# 	y_pred = []
		# 	y_true = []
		# 	y_pred_tmp = []
		# 	per = solution.per
		# 	with concurrent.futures.ProcessPoolExecutor() as executor:
		# 		for data, target in test_loader:
		# 			y_true.append(np.array(target).reshape(-1))
		# 			y_pred_tmp.append(data)
		# 			print("1111")
		# 			if len(y_pred_tmp) >= per:
		# 				mystarttime = time.time()
		# 				y_pred_tmp = executor.map(solution.inference, y_pred_tmp)
		# 				duration += time.time() - mystarttime
		# 				y_pred_tmp = np.concatenate([i for i in y_pred_tmp])
		# 				y_pred.append(y_pred_tmp)
		# 				y_pred_tmp = []
		# 				print("2222")
				
		# 		if len(y_pred_tmp) > 0:
		# 			mystarttime = time.time()
		# 			y_pred_tmp = executor.map(solution.inference, y_pred_tmp)
		# 			duration += time.time() - mystarttime
		# 			y_pred_tmp = np.concatenate([i for i in y_pred_tmp])
		# 			y_pred.append(y_pred_tmp)
		# 			y_pred_tmp = []

		# 	print(f"Inference parallel took {duration:.4f} seconds")
		else:
			y_true, y_pred = [], []
			print("begin inference...")
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				y_true.append(np.array(target).reshape(-1))
				mystarttime = time.time()
				out = solution.inference(data)
				duration += time.time() - mystarttime
				y_pred.append(out)
			print(f"Inference took {duration:.4f} seconds")
		y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)



	cpu_time = (time.time() - start_time)
	metric_dict = classification_metrics(y_true, y_pred)
	spe = metric_dict['spe']
	kappa = metric_dict['kappa']
	f1 = metric_dict['f1']

	score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
	score_file.write(f"Kappa: {kappa}\n")
	score_file.write(f"Macro_F1: {f1}\n")
	score_file.write(f"Macro_Specificity: {spe}\n")
	score_file.write(f"CPU_Time: {cpu_time}\n")
	score_file.close()
	