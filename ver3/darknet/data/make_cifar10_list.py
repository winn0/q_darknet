import os
cwd = os.getcwd()
test_dir = cwd+'/test'
train_dir = cwd+'/train'


test_f = open(cwd+"/cifar/test.list", 'w')
for (root, directories, files) in os.walk(test_dir):
    for file in files:
        file_path = os.path.join(root, file)
        test_f.write(file_path+"\n")
test_f.close()    
train_f = open(cwd+"/cifar/train.list", 'w')
for (root, directories, files) in os.walk(train_dir):
    for file in files:
        file_path = os.path.join(root, file)
        train_f.write(file_path+"\n")
train_f.close()    