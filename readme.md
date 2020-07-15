**result5** :
	以CycleGAN数据作为输入，对数据进行了norm_channel;
	在epoch为64时，训练集和测试集准确率骤降至50%；
**result6**：
        以CycleGAN数据作为输入，不进行norm_channel；
	~~在epoch为10时，准确率骤降;~~(是因为学习率发生了改变)
	正确率正常

>[2020-07-13 20:30:45.273 CST ~~ INFO    ] Epoch   189 train_loss: 0.145566347547, train_acc: 0.966044776119, val_loss: 0.237934307406, val_acc: 0.938988095238, test_loss: 3.14534151822, test_acc: 0.774621212121

**result7**：
	在6的基础上，使用reset model；
	start-filter：5
	num-blocks:5
	训练很慢，开始一直保持在0.5

-----------------------------

resultAutoGan:

  AutoGan数据，不进行norm_channel;

训练集horse， 测试集apple

**结果**：

[2020-07-14 13:27:41.405 CST ~~ INFO    ] Epoch   200 train_loss: 0.66534659994, train_acc: 0.68544600939, val_loss: 0.649774422623, val_acc: 0.64858490566, test_loss: 0.950040320175, test_acc: 0.633064516129

-------------------

resultAutoGan2：

 将` O = Dense(128,  activation='relu')(O)`改为`O = Dense(64,  activation='relu')(O)`；

无论增减全连接个数，抑或增加层数，效果不显著；

将全连接层删除，效果有提升；但是到了epoch=10时，学习率从0.01变至0.1(是因为代码中schedule函数，针对resnet进行了学习率的调整)

将全连接层删除后，可以将batch_size从8提升至16；

` python scripts/run.py train -w ./resultAutoGan2 --dataset others --model complex --batch-size 16 --schedule null`