import random
from util import *


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
    bot = BotStarter()

    parser = BotParser(bot)
    parser.run()


if __name__ == '__main__':
    go()
