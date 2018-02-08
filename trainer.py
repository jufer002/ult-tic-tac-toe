import tensorflow as tf
import numpy as np

from util import *

import json
import subprocess
import random
import os


# Tensorflow preallocates a percentage of 
# GPU memory, which causes errors when
# multiple processes are introduced.
# This configuration makes it so memory
# is not preallocated.
def get_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return config


# Dictionary for massaging string boards into float boards for network input.
board_to_float_map = { 
    '0' : -0.5,
    '.' : 0.0,
    '1' : 0.5,
}

# Utility functions.

# Converts JSON result data into all relevant board states.
def json_data_to_fields(data):
    # Gather the logs which contain sequential board states.
    bot_1_log = data['players'][0]['log'].split('\n')
    bot_2_log = data['players'][1]['log'].split('\n')

    # Obtain all microboards in sequential order, [3:] cuts out the message 'update game field'
    # from the line.
    return ([line.split(' ')[3:] 
                for line in bot_1_log 
                if 'update game field' in line],
            [line.split(' ')[3:] 
                for line in bot_2_log 
                if 'update game field' in line])

# Loads wiunning bot ID from JSON data.
def json_data_to_winner(data):
    return json.loads(data['details'])['winner']

# Takes a resultfile and returns board states for both bots and winner.
def load_from_json(resultFile):
    # Load the JSON result file into memory.
    with open(resultFile) as f:
        data = json.load(f)

    fields_0, fields_1 = json_data_to_fields(data)

    winner = json_data_to_winner(data)

    return (json_fields_to_floats(fields_0), 
            json_fields_to_floats(fields_1), 
            int(winner))

def json_fields_to_floats(fields):
    board = [[[board_to_float_map[spot]]
                    for board in field
                    for spot in board.split(',')] for field in fields[:81]]
    
    return board

def load_moves(bot_id):
    # Load JSON bad moves file into memory.
    with open('moves-' + str(bot_id) + '.json') as f:
        data = json.load(f)

    bad_moves, good_moves, actions, best_move_good = data['illegal_moves'], data['available_moves'], data['actions'], data['best_move_good']
    
    good_moves = [[Move(move[0], move[1]) for move in state_moves] for state_moves in good_moves]

    return bad_moves, good_moves, actions, best_move_good

def run_stats():
    subprocess.Popen('~/run_stats.sh',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

class QTrainer:
    # Runs UTT and trains Q network.
    def __init__(self, learning_rate=0.1, gamma=0.9):
        # The learning rate is used in backprop.
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Create buffer to store games in.
        self.games = {}

        # Create q table where every element is of the form (s1, a, r, s2)
        self.q_table = []

        # Loss values for bots 0 and 1, used to make sure the loss is decreasing.
        self.loss_0 = []
        self.loss_1 = []

    def save_weights(self, sess, bot_id):
        self.saver.save(sess, 'weights-' + str(bot_id) + '/weights.ckpt')

    def load_weights(self, sess, bot_id):
        self.saver.restore(sess, 'weights-' + str(bot_id) + '/weights.ckpt')

    # Build the network for a given bot.
    def build_network(self, bot_id):
        tf.reset_default_graph()

        # First 81 inputs are board, last 81 are bit mask (1 if playable else 0)
        self.in_layer = tf.placeholder(shape=[1, 81 * 2], dtype=tf.float32, name='input_layer')

        # Load the weights
        self.W1 = tf.Variable( tf.random_uniform([81 * 2, 128]), name='bot_' + str(bot_id) + '_weights_1')
        self.B1 = tf.Variable( tf.random_uniform([128]), name='bot_' + str(bot_id) + '_bias_1')

        self.W2 = tf.Variable( tf.random_uniform([128, 64]), name='bot_' + str(bot_id) + '_weights_2')
        self.B2 = tf.Variable( tf.random_uniform([64]), name='bot_' + str(bot_id) + '_bias_2')

        self.W3 = tf.Variable( tf.random_uniform([64, 32]), name='bot_' + str(bot_id) + '_weights_3')
        self.B3 = tf.Variable( tf.random_uniform([32]), name='bot_' + str(bot_id) + '_bias_3')

        self.W4 = tf.Variable( tf.random_uniform([32, 81]), name='bot_' + str(bot_id) + '_weights_4')
        self.B4 = tf.Variable( tf.random_uniform([81]), name='bot_' + str(bot_id) + '_bias_4')

        # Create hidden layer operations
        hidden1 = tf.nn.relu( tf.matmul(self.in_layer, self.W1) + self.B1, name='hidden_layer_1')
        hidden2 = tf.nn.relu( tf.matmul(hidden1, self.W2) + self.B2, name='hidden_layer_2')
        hidden3 = tf.nn.relu( tf.matmul(hidden2, self.W3) + self.B3, name='hidden_layer_3')

        # Create output layer
        self.output_layer = tf.nn.relu( tf.matmul(hidden3, self.W4) + self.B4, name='output_layer')

        # Find the highest reward in the output dimension of the output layer.
        self.max_reward = tf.reduce_max( self.output_layer, reduction_indices=[1] )

    def build_training_procedure(self):
        # Action a
        self.output_index = tf.argmax( tf.gather(self.output_layer, 0) )

        self.action_holder = tf.placeholder(shape=(), dtype=tf.int32)

        self.s1_output = self.output_layer[0][self.action_holder]
        
        # Feed actual reward r into here.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        # Estimate for q function given by feeding s1 through network.
        self.Q_estimate = tf.placeholder(shape=[None], dtype=tf.float32)

        # Calculate loss by finding MSE of the network's estimate for Q function and 'actual' Q function
        self.loss = tf.losses.mean_squared_error(self.reward_holder + self.max_reward * self.gamma, self.Q_estimate)

        # Create training operation that minimizes the loss function - applies backpropagation.
        self.train_operation = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # Create operation to save weights to a file.
        self.saver = tf.train.Saver()

    def get_loss(self):
        return 'loss 0: ' + str(self.loss_0) + '\nloss 1: ' + str(self.loss_1) + '\n'

    def play_game(self):
        subprocess.Popen('./run.sh', 
            shell=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL).wait()

    # Creates an empty table for the two bots to store their states and actual rewards.
    def clear_buffers(self, epoch):
        self.games[epoch] = {
            # For bot 0.
            0: 
                {'state_buffer': [], 
                'reward_buffer': [],
                'best_move_good': [],},
            # For bot 1.
            1:
                {'state_buffer': [],
                'reward_buffer': [],
                'best_move_good': [],}
        }

    def write_q_table(self):
        with open('q_table.json', 'w') as f:
            json.dump(self.q_table, f)

    # Reward the winner and punish the loser.
    def reward_winner(self, winner, epoch):
        bot_0_reward = 10.0 if winner == 0 else -10.0
        bot_1_reward = 10.0 if winner == 1 else -10.0

        self.games[epoch][0]['reward_buffer'][-1] = bot_0_reward
        self.games[epoch][1]['reward_buffer'][-1] = bot_1_reward

    def determine_rewards(self, bad_moves_0, bad_moves_1, current_round):
        if current_round in bad_moves_0:
            reward_0 = -10.0
        else:
            reward_0 = 0

        if current_round in bad_moves_1:
            reward_1 = -10.0
        else:
            reward_1 = 0

        return (reward_0, reward_1)

    def index_to_move(self, index):
        board_width = 9

        # These equations map the bot's 1D index onto the 2D board.
        x = index % board_width
        y = index // board_width

        return Move(x, y)

    def preprocess(self, fields, good_moves):
        for i in range(len(fields)):
            for j in range(len(fields[i])):
                move = self.index_to_move(j)
                if i >= len(good_moves):
                    fields[i].append([0.0])
                elif move in good_moves[i]:
                    fields[i][j][0] = 10.0
                    fields[i].append([1.0])
                else:
                    fields[i].append([0.0])

        return fields

    # Load all games in games.json into memory.
    def load_q_table(self):
        '''
        with open('games.json', 'r') as f:
            self.games = json.load(f)

        self.games = { int(k) : self.games[k] for k in self.games.keys() }

        for i in range(1, len(self.games) + 1):
            self.games[i] = { int(k) : self.games[i][k] for k in self.games[i].keys() }
        '''
        with open('q_table.json', 'r') as f:
            self.q_table = json.load(f)
        

    # Load the most recent game into memory.
    def load_recent_game(self, epoch):
        self.clear_buffers(epoch)

        # Load all the fields and the winner from the game into memory.
        fields_0, fields_1, winner = load_from_json('resultfile.json')

        bad_moves_0, good_moves_0, actions_0, best_move_good_0 = load_moves(0)
        bad_moves_1, good_moves_1, actions_1, best_move_good_1 = load_moves(1)

        states_0, states_1 = self.preprocess(fields_0, good_moves_0), self.preprocess(fields_1, good_moves_1)

        bot_0_s1 = states_0[0]
        bot_1_s1 = states_1[1]

        action_counter = 0

        a_0 = actions_0[action_counter]
        a_1 = actions_1[action_counter]

        current_round = 0
        for state_0, state_1 in zip(states_0[1:], states_1[1:]):
            self.games[epoch][0]['state_buffer'].append(state_0)
            self.games[epoch][1]['state_buffer'].append(state_1)

            reward_0, reward_1 = self.determine_rewards(bad_moves_0, bad_moves_1, current_round)

            self.q_table.append( (bot_0_s1, a_0, reward_0, state_0) )
            self.q_table.append( (bot_1_s1, a_1, reward_1, state_1) )

            bot_0_s1 = state_0
            bot_1_s1 = state_1

            action_counter += 1

            # If the game is over, don't try to get the next action.
            if action_counter < len(actions_0):
                a_0 = actions_0[action_counter]
            if action_counter < len(actions_1):
                a_1 = actions_1[action_counter]

            self.games[epoch][0]['reward_buffer'].append(reward_0)
            self.games[epoch][1]['reward_buffer'].append(reward_1)

            self.games[epoch][0]['best_move_good'].append(best_move_good_0)
            self.games[epoch][1]['best_move_good'].append(best_move_good_1)

            current_round += 1

        self.reward_winner(winner, epoch)

        return winner

    def replay_experience(self, epoch, train_cycle):
        # Create function to get some random games.
        total_states = len(self.q_table) - 1
        get_states = lambda: [random.randint(0, total_states) for _ in range(train_cycle)]

        # Get states for bots
        states_0, states_1 = get_states(), get_states()

        for state_0, state_1 in zip(states_0, states_1):
            # Update weights in bot 0.
            self.bot_learn(0, state_0)

            # Update weights in bot 1.
            self.bot_learn(1, state_1)

        tf.reset_default_graph()

    def store_loss(self, sess, bot_id, loss):
        if bot_id == 0:
            self.loss_0.append(loss)
        if bot_id == 1:
            self.loss_1.append(loss)

    def bot_learn(self, bot_id, row):
        # Build the network for the given bot.
        self.build_network(bot_id)
        # Build the training procedure for the given bot.
        self.build_training_procedure()

        config = get_tf_config()

        init = tf.global_variables_initializer()
        # Learn.
        with tf.Session(config=config) as sess:
            sess.run(init)

            self.load_weights(sess, bot_id)

            s1, a, r, s2 = self.q_table[row]

            # Feed s1 into network.
            s1_estimate = sess.run(self.s1_output,
                feed_dict={
                    self.in_layer: np.transpose(s1),
                    self.action_holder: a,
                })

            # Feed s2 into network and train.
            sess.run(self.train_operation,
                feed_dict={
                    self.in_layer: np.transpose(s2),
                    self.Q_estimate: [float(s1_estimate)],
                    self.reward_holder: [float(r)],
                })

            # Calculate loss.
            loss = sess.run(self.loss,
                    feed_dict={
                        self.in_layer: np.transpose(s2),
                        self.Q_estimate: [float(s1_estimate)],
                        self.reward_holder: [float(r)],
                    })

            self.store_loss(sess, bot_id, loss)

            self.save_weights(sess, bot_id)


# Main function. Trains bots!
def train_bots(epochs, train_cycle, stats=False):
    trainer = QTrainer()

    if os.path.isfile('q_table.json'):
        trainer.load_q_table()

    for epoch in range(1, epochs + 1):
        # Have two bots play against each other.
        trainer.play_game()

        # Load the game states and rewards into buffers.
        trainer.load_recent_game(epoch)

        if epoch % train_cycle == 0:
            # Train the bots with experience replay.
            trainer.replay_experience(epoch, train_cycle * 10)

            with open('train_log.txt', 'w') as f:
                f.write('training on epoch ' + str(epoch) + '\n')
                f.write(trainer.get_loss())

            if stats:
                run_stats()

            trainer.write_q_table()

def train_on_games(epochs):
    trainer = QTrainer()
    trainer.load_q_table()

    num_games = len(trainer.games)

    trainer.replay_experience(num_games, epochs)
    with open('train_log.txt', 'w') as f:
        f.write('training on epoch ' + str(epochs) + '\n')
        f.write(trainer.get_loss())

    run_stats()

def test():
    trainer = QTrainer()
    
    trainer.play_game()
    trainer.load_recent_game(1)

    trainer.replay_experience(1, 1)

    with open('train_log.txt', 'w') as f:
        f.write('training on epoch ' + str(1) + '\n')
        f.write(trainer.get_loss())
    
    trainer.write_q_table()

if __name__ == '__main__':
    train_bots(epochs=50000, train_cycle=500, stats=True)

