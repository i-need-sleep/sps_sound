import numpy as np

name_root = 'encode_test_set_scale_singleInst_1dim_betavae'
metric = 'linearProjectionMSE'
# metric = 'R2'
nums = []

for i in range(10):
    if i == 8:
        continue
    path = f'{name_root}_{str(i)}_{metric}.txt'
    with open(path, 'r') as f:
        data = f.read()
    data = data.split(', ')[1]
    nums.append(float(data))
print(nums)
nums = np.array(nums)
print(np.average(nums))
print(np.std(nums))