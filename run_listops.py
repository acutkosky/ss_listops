
import torch
import listops_trainer
import listops_model
import pickle


model_config = listops_model.ModelConfig(d_state=512, scaling=0.001)

train_config = listops_trainer.TrainerConfig(lr=0.0001)



f = open('save_listops.pkl','rb')
listops = pickle.load(f)
f.close()



trainer = listops_trainer.Trainer(train_config, model_config, listops=listops)


trainer.train(50)
