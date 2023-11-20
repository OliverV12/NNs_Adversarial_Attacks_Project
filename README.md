##############################################################################
### GENERATING ADVERSARIAL ATTACKS ON MNIST DATASET VIA AUTOATTACK LIBRARY ###
##############################################################################

INFORMATION ON THE CODE STRUCTURE AND FILES/FOLDER ORGANIZATION
The project's code is organized into four Python modules: "Adv_attack", "Adv_attack_my_func", "Backbones" and "classifier".
The main program is located in the "Adv_attack" file while "Adv_attack_my_func", "Backbones" and "classifier" contain, respectively, functions implemented by me, three distinct classes (neural network architectures) named 'MLP1,' 'MLP2,' and 'SimpleCNN, and the "Classifier" class (where 'resnet18' is also defined). 

"Adv_attack_my_func", "Backbones" and "classifier" are externally imported into "Adv_attack" during code execution.

------------------------------------------------------------------------------------------------------------------------------------------

HOW TO RUN THE CODE
In "Adv_attack", as explained in more detail in the slides, the main program consists of four IF statement blocks: 'train', 'eval', 'adv_attacks_generation' and 'eval_other_adv_attacks', executable only one at a time and all via terminal.

- To execute the 'train' block:
  	Example:
		type: 'python Adv_attack.py train --backbone resnet18 --batch_size 64  --epochs 3' 
	For "--backbone," you can select from 'mlp1,' 'mlp2,' 'simplecnn,' and 'resnet18.'

- To execute the 'eval' block:
	Example:
		type: python Adv_attack.py eval --backbone mlp1 --batch_size 64

- To execute the 'adv_attacks_generation' block:
	Example:
		type: python Adv_attack.py adv_attacks_generation --backbone mlp2

- To execute the 'eval_other_adv_attacks' block:
	Example:
		type: python Adv_attack.py adv_attacks_generation --backbone simplecnn

At the end of the execution of the 'train' and 'adv_attacks_generation' blocks, the files related to the architecture of the trained model will be saved.
  
		  


