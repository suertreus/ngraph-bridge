import pdb, re
flname = 'log_puretf_1.txt'

oplist = []
for line in open(flname):
    if "kernel done" in line:
        tmp = line.split("kernel done:")[-1]
        result = re.search('{{(.*)}}', tmp)
        opinfo = result.group(1)
        if "Async" in line:
            oplist.append('---->async: ' + opinfo)
        else:
            oplist.append('sync: ' + opinfo)

pdb.set_trace()
for idx, ln in enumerate(oplist):
    if '_retval_ConvNet' in ln or 'async' in ln or 'Cast' in ln or 'IteratorV2' in ln or 'Make' in ln or 'ModelDat' in ln or 'IteratorToStringHandle' in ln : # or 'Mean_1' in ln or 'truediv' in ln:
        print(idx, ln)


'''
2 sync: node datasets/iterator/IteratorV2
39 ---->async: node _recv_Placeholder/_0_0
40 ---->async: node _recv_Placeholder/_1_0
52 sync: node datasets/iterator/ModelDataset
53 sync: node datasets/iterator/MakeIterator


123 sync: node datasets/iterator/IteratorV2
165 sync: node preprocess/Mean_1 X lots
312 ---->async: node datasets/X_Y/IteratorGetNext
354 sync: node preprocess/Mean_1 X lots
1816 sync: node ConvNet/Cast
1822 sync: node _retval_ConvNet/Mean_0_0
1848 sync: node datasets/iterator/IteratorV2
1860 ---->async: node datasets/X_Y/IteratorGetNext
1869 sync: node preprocess/Mean_1
1876 sync: node preprocess/Mean_1
1878 sync: node preprocess/Mean_1
1886 sync: node preprocess/Mean_1
1919 sync: node preprocess/Mean_1
1922 sync: node preprocess/Mean_1
'''