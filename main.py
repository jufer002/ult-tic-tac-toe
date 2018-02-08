#!/usr/bin/env python3

import sys
import random
from util import *

import tensorflow as tf
import numpy as np
import os
import json


# Tensorflow preallocates a percentage of 
# GPU memory, which causes errors when
# multiple processes are introduced.
# This configuration makes it so memory
# is not preallocated.
def get_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return config

class LearnBot:
    '''
    Learn to play Ultimate-tic-tac toe with a 
    deep reinforcement learning neural network.
    '''

    # Maps game board symbols to floats for network input
    __board_to_float_map = { 
        '0' : -0.5,
        '.' : 0.0,
        '1' : 0.5,
    }

    # Initialize network
    def __init__(self, write_out=False):
        # This is the bot's UTT engine ID. It should be set
        # once a parser has been made so that it can know to train
        # for X winning or O winning.
        self.id = None

        # Is the bot allowed to write to a file?
        self.write_out = write_out

        # Keep track of what round the bot is on.
        self.round = 0

        # Keep track of all the bot's illegal moves to discourage it during
        # training, and also of available moves.
        self.moves = {'available_moves': [], 'illegal_moves': [], 'best_move_good': [], 'actions': [],}

    # Write message to file.
    def write_msg(self, message):
        if not self.write_out:
            return

        orig = sys.stdout

        with open('bot_messages-' + str(self.id), 'a') as f:
            f.write(str(message) + '\n')

        sys.stdout = orig

    # Load weights from file.
    def load_weights(self, sess):
        if self.id is None:
            return

        orig = sys.stdout
        self.saver.restore(sess, 'weights-' + str(self.id) + '/weights.ckpt')
        sys.stdout = orig

    # Save weights to a file
    def save_weights(self, sess):
        if self.id is None:
            return

        orig = sys.stdout
        self.saver.save(sess, 'weights-' + str(self.id) + '/weights.ckpt')
        sys.stdout = orig

    # Let the bot know which player it is. X, or O (actually 0 or 1)
    def set_id(self, id):
        self.id = id

        self.write_msg('I am player ' + str(id))

        # Build the network and setup cost functions and backpropagation.
        self.build_network()

    def write_moves(self):
        orig = sys.stdout
        with open('moves-' + str(self.id) + '.json', 'w') as f:
            json.dump(self.moves, f)

        sys.stdout = orig

    # Create the network as a computational graph in Tensorflow
    def build_network(self):
        # Tensorflow builds a computational graph, make sure it's empty before the network is built.
        tf.reset_default_graph()

        # Create placeholder for UTT board as input
        self.in_layer = tf.placeholder(shape=[1, 81 * 2], dtype=tf.float32, name='input_layer')

        # Set up identifier to prepend to weight names,
        # so the trainer knows which weights to train.
        weight_id = 'bot_' + str(self.id) + '_'

        # Create 3 weight tensors for the 2 hidden layers, with corresponding biases.
        self.W1 = tf.Variable( tf.random_uniform([81 * 2, 128]), name=weight_id + 'weights_1')
        self.B1 = tf.Variable( tf.random_uniform([128]), name=weight_id + 'bias_1')

        self.W2 = tf.Variable( tf.random_uniform([128, 64]), name=weight_id + 'weights_2')
        self.B2 = tf.Variable( tf.random_uniform([64]), name=weight_id + 'bias_2')

        self.W3 = tf.Variable( tf.random_uniform([64, 32]), name=weight_id + 'weights_3')
        self.B3 = tf.Variable( tf.random_uniform([32]), name=weight_id + 'bias_3')

        self.W4 = tf.Variable( tf.random_uniform([32, 81]), name=weight_id + 'weights_4')
        self.B4 = tf.Variable( tf.random_uniform([81]), name=weight_id + 'bias_4')

        # Create hidden layer operations
        hidden1 = tf.nn.relu( tf.matmul(self.in_layer, self.W1) + self.B1, name='hidden_layer_1')
        hidden2 = tf.nn.relu( tf.matmul(hidden1, self.W2) + self.B2, name='hidden_layer_2')
        hidden3 = tf.nn.relu( tf.matmul(hidden2, self.W3) + self.B3, name='hidden_layer_3')

        # Create output layer
        self.output_layer = tf.nn.relu( tf.matmul(hidden3, self.W4) + self.B4, name='output_layer')
        
        # Find the highest reward in the output dimension of the output layer.
        #self.predicted_reward = tf.reduce_max( self.output_layer, reduction_indices=[1] )
        
        # Gather the output dimension of the output layer and return the index with the highest value.
        #self.output_index = tf.argmax( tf.gather(self.output_layer, 0) )

        # Create operation to save weights to a file and load them from a file.
        self.saver = tf.train.Saver()


    def get_best_move(self, predicted_rewards, state):
        available_moves = state.getField().getAvailableMoves()
        # If there are no available moves, just give up.
        if not available_moves:
            return None

        legal_indices_and_rewards = [ (i, reward) for i, reward in enumerate(predicted_rewards)
                                    if self.index_to_move(state, i) in available_moves ]

        max_index, max_reward = legal_indices_and_rewards[0][0], legal_indices_and_rewards[0][1]

        for i, reward in legal_indices_and_rewards:
            if reward > max_reward:
                max_reward = reward
                max_index = i

        self.moves['best_move_good'].append(str(float(reward) == float(max(predicted_rewards))))

        return int(max_index)

    def feed_forward(self, input_board, state):
        config = get_tf_config()

        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)

            if os.path.exists('weights-' + str(self.id)):
                self.load_weights(sess)
            else:
                self.save_weights(sess)

            predicted_rewards = np.hstack(sess.run(self.output_layer, feed_dict={self.in_layer: input_board}))

            best_choice = self.get_best_move(predicted_rewards, state)

        return best_choice

    def favor_available_moves(self, state, board):
        # Gather available moves.
        available_moves = state.getField().getAvailableMoves()

        self.moves['available_moves'].append([(move.getX(), move.getY()) for move in available_moves])

        # Check all moves to see if they are available.
        for i in range(len(board[0])):
            move = self.index_to_move(state, i)
            if move in available_moves:
                board[0][i] = 1.0
                board[0].append(1.0)
            else:
                board[0].append(0.0)

        return board

    def state_to_board(self, state):
        field = state.getField().getMicroboard()

        # Transform all board spots to floats using the dictionary
        board = [[self.__board_to_float_map[spot]
                    for row in field 
                    for spot in row]]

        return self.favor_available_moves(state, board)

    def index_to_move(self, state, index):
        board_width = state.getField().getNrColumns()

        # These equations map the bot's 1D index onto the 2D board.
        x = index % board_width
        y = index // board_width

        return Move(x, y)

    def move_to_index(self, move):
        x = move.getX()
        y = move.getY()

        return x + y * 9

    def doMove(self, state):
        # Massage board from state into matrix of floats of shape [1, 162]
        input_board = self.state_to_board(state)

        # Feed board into network
        chosen_index = self.feed_forward(input_board, state)
        if chosen_index is None:
            return None

        # Turn network output into move
        move = self.index_to_move(state, chosen_index)

        legal = state.isLegalMove(move)

        # If the bot makes an illegal move, have it act legally and randomly and 
        # give it a negative reward.
        if not legal:
            move = self.do_random_move(state)
            self.moves['illegal_moves'].append(self.round)

        if np.random.randint(100) < 11:
            move = self.do_random_move(state)

        self.write_moves()

        self.round += 1

        self.moves['actions'].append( self.move_to_index(move) )
        return move

    def do_random_move(self, state):
        self.write_msg('I am moving randomly.')
        moves = state.getField().getAvailableMoves()
        if (len(moves) > 0):
            return moves[random.randrange(len(moves))]
        else:
            return None


class BotStarter:
    def __init__(self):
        random.seed()
        
    def doMove(self, state):
        moves = state.getField().getAvailableMoves()

        if (len(moves) > 0):
            return moves[random.randrange(len(moves))]
        else:
            return None

def go():
    bot = LearnBot()

    parser = BotParser(bot)
    parser.run()


if __name__ == '__main__':
    go()
