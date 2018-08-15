# Super AIsaac Game Agent
My PPO agent implementation for The Binding of Isaac: Afterbirth+ in the SerpentAI Monstro Boss Fight environment.
See SerpentSuperAIsaacGamePlugin for the GamePlugin used for the environment.

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
Modeled each action space separately. More on this in Rewards section.  

## Rewards
I found that it made sense to explicitly model the multiple action dimensions for Isaac by splitting up attack and movement actions. This results in the ability to optimize these policies separately from the same shared model, and also permits separate rewards for these different action dimensions. The following rules were used for shaping the raw rewards.

English description of rewards:

movement is penalized when damage is taken or Isaac dies, while attack is never penalized  
no attack reward if damage is taken during same observation  
if nothing happens then movement is rewarded a little for staying alive  
Movement and attack are both rewarded when boss damage is done or boss is killed  

Exact equation can be found in super_ml.super_reward

## Hyperparameters 
Hyperprarameters are often very difficult to determine. In this section I will provide justification for why I chose my model's hyperparameters in order to provide insight for others looking to make changes or adapt the architecture for a different environment. 

### Learning Rate
learning_rate = 5e-5  
Original PPO paper used 1e-4, this model trained with 1e-4 for the first ~150k steps, and then decayed that learning rate to 5e-5.  
Relatively robust, would recommend anywhere from 1e-4 to 1e-5, with some benefit from some linear or scheduled decay to a lower value.  

### Momentum
momentum = 0.5  
This is the momentum used with ADAM. No tests were made with alternative momentum values, but the justification was that the parameter updates in a RL environment may shift much quicker in a RL context, which would justify using a lower momentum value. 

### Gamma
gamma = 0.75  
This is a significantly lower discount future rewards decay rate. This parameter required significant trial-and-error, where a lower value tended to result in a model which prioritized reactive actions, which makes sense in an Isaac-like game where you must dodge and attack in real-time. One interpretation of gamma is as a "Probability of survival" for every time-step, and, while the model does not have a 75% chance of death at every action, it definitely has a high probability of death if it is unable to react to very close-in-time attacks from a boss. 

Additionally, in my opinion, it is likely very difficult, if not nearly impossible, for a model which only uses the last few frames as input to be able to learn a value function for discount future rewards if this gamma decay value is too large because the model will not have a historical understanding of the progress made and progress left to make in a boss fight. Consider a larger gamma when a RNN layer is added so that the model can consider more long-term movement and attack strategies. 

### Move Entropy Scale
move_entropy_scale = 5e-3  
This draws movement action probabilities towards a uniform distribution, which helps the model explore and not exploit too early on in training.

### Attack Entropy Scale
attack_entropy_scale = 5e-3  
This draws attack action probabilities towards a uniform distribution, which helps the model explore and not exploit too early on in training.

### Surrogate Objective Clip for PPO
surrogate_objective_clip = 0.2  
This represents the acceptable ratio difference between a previous action probability and an updated action probability, as described by PPO. Consider reducing this to 0.1 if a larger learning rate or a larger memory capacity is used.

### Value Loss Coefficient
value_loss_coefficient = 0.5  
This represents how much the value function loss contributes towards the model's combined loss function. The previous paper utilized 0.5-1.0, so I would recommend somewhere in that range. I found no significant difference between 1.0 and 0.5, so I left it at 0.5.

### Batch Size 
batch_size = 64  
This influences your mini-batch size for parameter updates. It can significantly influence your loss surface during training. I would recommend sticking within a 32-128 range, as smaller batches have been known to improve generalization due to regularizing noise, but smaller batches are also worse estimates of the policy gradient. Smaller batch sizes will also result in more parameter updates per memory capacity being reached.

### Memory Capacity
memory_capacity = 4096  
This controls how many observations you will collect before training on the observed data. 2048-8192 are relatively good values for this. Consider increasing if average episode length increases, as that means more of the observations will be from the same few episodes. The larger the memory capacity, the more accurate the estimated policy gradient updates, but also much more episodes per parameter update will need to be collected, along with potential off-policy clipping occurring due to so many parameter updates. Be sure to monitor the PPO clipping ratio to make sure you are not clipping too many updates, as that may result in poor parameter updates and it may also drive your policies more towards uniform due to your entropy loss always being applied for every parameter update.

### Epochs
epochs = 2  
This controls how many times you will iterate over the collected memory capacity observations. If you increase your memory capacity, keep in mind that you will increase the parameter updates by a factor of the number of epochs, so you may want to reduce epochs if you increase memory capacity in order to maintain a similar number of parameter updates per episode.

## SerpentAI
SerpentAI was used to manage and interact with the game environment.
Check out SerpentAI here: [https://github.com/SerpentAI/SerpentAI](https://github.com/SerpentAI/SerpentAI)

## TensorFlow
TensorFlow was used for neural network layers and computation. 
Check out TensorFlow here: [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
