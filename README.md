# Super AIsaac Game Agent
My PPO implementation for The Binding of Isaac: Afterbirth+ in the SerpentAI Monstro Boss Fight environment

## Gameplay
Check my stream for live gameplay or historical VODs while training.
Stream:
[https://www.twitch.tv/supermaxman](https://www.twitch.tv/supermaxman)

VODs:
[https://www.twitch.tv/supermaxman/videos](https://www.twitch.tv/supermaxman/videos)

## Architecture
Input provides four grayscale game image frames of size 120x68
Relu activation function applied after every convolution
Convolution2D(filters=64, kernel_size=8, strides=4, padding='valid')
Convolution2D(filters=128, kernel_size=4, strides=2, padding='same')
Convolution2D(filters=256, kernel_size=4, strides=2, padding='same')
Convolution2D(filters=512, kernel_size=4, strides=2, padding='same')
Convolution2D(filters=1024, kernel_size=3, strides=1, padding='same')
Fully connected layers used from flattened output of last convolutional layer to action probabilities and values.
Modeled each action space seperately. More on this in Rewards section.

## Rewards
I found that it made sense to explicitly model the multiple action dimensions for Isaac by splitting up attack and movement actions.
This results in the ability to optimize these policies seperately from the same shared model, and also permits seperate rewards
for these different action dimensions. The following rules were used for shaping the raw rewards.

English description of rewards:
movement is penalized when damage is taken or isaac dies, while attack is never penalized
no attack reward if damage is taken during same observation
if nothing happens then movement is rewarded a little for staying alive
Movement and attack are both rewarded when boss damage is done or boss is killed

Exact equation is as follows:
move_reward = 0.0
attack_reward = 0.0
if isaac_is_alive:
    if isaac_took_damage:
        move_reward += -1.0 * damage_taken
    else:
        if damage_dealt_to_boss:
            move_reward += 0.05
            attack_reward += 0.05
        else:
            move_reward += 0.01
    if boss_is_dead:
        move_reward += 1.0
        attack_reward += 1.0
else:
    move_reward += -2.0

## Hyperparameters 
Hyperprarameters are often very difficult to determine. In this section I will provide justification for why I chose my model's hyperparameters in order to provide insight for others looking to make changes or adapt the architecture for a different environment. 

### Learning Rate
learning_rate = 5e-5

### Momentum
momentum = 0.5

### Gamma
gamma = 0.75

### Move Entropy Scale
move_entropy_scale = 5e-3

### Attack Entropy Scale
attack_entropy_scale = 5e-3

### Surrogate Objective Clip for PPO
surrogate_objective_clip = 0.2

### Value Loss Coefficient
value_loss_coefficient = 0.5

### Batch Size 
batch_size = 64

### Memory Capacity
memory_capacity = 4096

### Epochs
epochs = 2

## SerpentAI
SerpentAI was used to manage and interact with the game environment.
Check out SerpentAI here: [https://github.com/SerpentAI/SerpentAI](https://github.com/SerpentAI/SerpentAI)

## TensorFlow
TensorFlow was used for neural network layers and computation. 
Check out TensorFlow here: [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
