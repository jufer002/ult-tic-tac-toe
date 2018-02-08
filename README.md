# ult-tic-tac-toe
This is a deep reinforcement learning project where I implemented Q-Learning to train a neural network to play Ultimate Tic Tac Toe.

Most of the interesting code is in trainer.py

trainer.py repeatedly runs the Java UTT program that spawns two bots who play against each other as specified in the wrapper commands json file.

trainer.py then reads the output files from the game and stores the moves and associated rewards in a big table.

Every train cycle, the trainer will use backpropagation to optimize for better rewards!
