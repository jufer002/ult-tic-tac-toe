from trainer import *


def write(data):
    with open('stats.txt', 'a') as f:
        f.write(str(data) + '\n')

# Return the % times that the QBot beats the starter bot.
# Assumes QBot is bot 0.
def starter_bot_test():
    trainer = QTrainer()

    q_wins = 0
    for epoch in range(100):
        trainer.play_game()

        _, _, winner = load_from_json('resultfile.json')
        if winner == 0:
            q_wins += 1
            write('The QBot won!')
        else:
            write('The starter bot won...')
    return q_wins

def run_tests():
    results = []


    results.append('Wins against starter bot: {}% of the time.'.format(starter_bot_test()))

    return results

if __name__ == '__main__':
    with open('train_log.txt', 'r') as f:
        epoch = f.readline().split(' ')[-1]

    print('testing Bot 0 on epoch {}'.format(epoch))
    write('on epoch {}'.format(epoch))
    write('the QBot...')

    [write(test) for test in run_tests()]
