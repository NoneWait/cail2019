import json
import random

with open('/home/delaiq/data/cail/big_train_data.json') as fh:
    data = json.load(fh)


casename = {}


for sample in data['data']:
    case = sample['paragraphs'][0]['casename']
    if case in casename:
        casename[case].append(sample)
    else:
        casename[case] = [sample]


for i in range(10):
    trainset = {}
    trainset['data'] = []
    trainset['version'] = data['version']

    devset = {}
    devset['data'] = []
    devset['version'] = data['version']
    for name, case in casename.items():
        # case_1
        dev_start = (len(case)//10) * i
        dev_end = (len(case)//10)*(i+1)

        devset['data'] += case[dev_start:dev_end]
        trainset['data'] += case[:dev_start]
        trainset['data'] += case[dev_end:]

    print(len(data['data']))
    print(len(trainset['data']))
    print(len(devset['data']))

    with open('/home/delaiq/data/cail/big_train_case_{}.json'.format(i), 'w', encoding="utf-8") as fh:
        json.dump(trainset, fh, ensure_ascii=False)

    with open('/home/delaiq/data/cail/big_dev_case_{}.json'.format(i), 'w', encoding="utf-8") as fh:
        json.dump(devset, fh, ensure_ascii=False)

    if i == 7:
        break

