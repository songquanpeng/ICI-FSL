import os
import requests
import time

dataset = "CIFAR100-FS"
saved_model = "trained/res12_cifar.pth.tar"

for classifier in ['lr', 'svm']:
    for num_shots, unlabeled_data_num in [(1, 0), (5, 0), (1, 15), (5, 15), (1, 30), (5, 50), (1, 80), (5, 80)]:
        command = f"python ./main.py --mode test --resume {saved_model} --dataset {dataset} --num_shots {num_shots} " \
                  f"--classifier {classifier} --unlabeled {unlabeled_data_num} --log_filename log/test_{dataset}.log"
        start_time = time.time()
        os.system(command)
        end_time = time.time()
        if 'MESSAGE_PUSH_URL' in os.environ:
            requests.get(f"{os.environ['MESSAGE_PUSH_URL']}Setting classifier={classifier} num_shots={num_shots} unlabeled_data_num={unlabeled_data_num} executed done with {(end_time - start_time) / 60} min used.")

# nums = input().split(); to_num = lambda x : float(x); nums = list(map(to_num, nums)); print(max(nums))
