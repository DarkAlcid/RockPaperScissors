import numpy as np
import matplotlib.pyplot as plt
import random
import os
import neat
import pickle
import visualize


moves = {
    'Rock' : 0, 
    'Paper' : 1, 
    'Scissors' : 2
}
idx_moves = {
    0 : 'Rock',
    1 : 'Paper',
    2 : 'Scissors'
}

mat_games = np.zeros((3,3))

mat_games[moves['Rock'], moves['Scissors']] = 1
mat_games[moves['Paper'], moves['Rock']] = 1
mat_games[moves['Scissors'], moves['Paper']] = 1
mat_games[moves['Rock'], moves['Paper']] = -1
mat_games[moves['Paper'], moves['Scissors']] = -1
mat_games[moves['Scissors'], moves['Rock']] = -1    

sequence_moves = [0, 1, 2, 1, 1, 2, 0, 0, 1, 2]
# sequence_moves = [0, 1, 2]
class Player :
    def __init__(self, move):
        self.move = move

    def play(self):
        self.move = random.randint(0, 2)

    def play_move(self, move):
        self.move = move

def eval_genomes(genomes, config):
    computer_old = [random.randint(0,2), random.randint(0,2), random.randint(0,2)]
    move_number = len(sequence_moves)
    score = 0

    nets = []
    ge = []
    nns = []

    for _, g in genomes :
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        nns.append(Player(0))
        g.fitness = 0
        ge.append(g)
    i = 0
    while True:
        i += 1
        computer_old.append(sequence_moves[(i-1) % move_number])
        computer = sequence_moves[i % move_number]
        computer_ohe = np.zeros((3,3))
        computer_ohe[0, computer_old[-1]] = 1
        computer_ohe[1, computer_old[-2]] = 1
        computer_ohe[2, computer_old[-3]] = 1
        if len(nns) == 0:
            break
        
        for x, nn in enumerate(nns):            
            # output = nets[x].activate((computer_old[-1], computer_old[-2], computer_old[-3]))
            output = nets[x].activate((
                computer_ohe[0, 0], computer_ohe[0, 1], computer_ohe[0, 2],
                computer_ohe[1, 0], computer_ohe[1, 1], computer_ohe[1, 2], 
                computer_ohe[2, 0], computer_ohe[2, 1], computer_ohe[2, 2] 
            ))
            
            move = np.argmax(output)
            nn.play_move(move)

            result = mat_games[nn.move, computer]

            ge[x].fitness += result

            if result == -1:
                ge[x].fitness += result * 10
                nns.pop(x)
                nets.pop(x)
                ge.pop(x)
            if result == 1:
                score += 1
                ge[x].fitness += result * 5

        if score > 100:
            break

def eval_genomes_with_player(genomes, config):
    player_old = [random.randint(0,2), random.randint(0,2), random.randint(0,2)]
    move_number = len(sequence_moves)
    score = 0

    nets = []
    ge = []
    nns = []

    for _, g in genomes :
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        nns.append(Player(0))
        g.fitness = 0
        ge.append(g)

    while True:
        player_move = player_move = int(input("Rock (0), Paper (1), Scissors(2) ?"))
        if player_move in range(2):
            player_old.append(player_move)
            if len(nns) == 0:
                break
            
            for x, nn in enumerate(nns):            
                output = nets[x].activate((player_old[-2], player_old[-3], player_old[-4]))

                move = np.argmax(output)
                nn.play_move(move)

                result = mat_games[nn.move, player_move]

                ge[x].fitness += result

                if result == -1:
                    ge[x].fitness += result * 10
                    nns.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                if result == 1:
                    score += 1
                    ge[x].fitness += result * 5

            if score > 100:
                break
        else :
            break
    

def training(config, winner_filename):
    # Create population based on the config we set
    p = neat.Population(config)

    # Give some outputs / information about each generation
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 100)
    pickle.dump(winner, open(winner_filename, 'wb'))
    node_names = {-1:'Rock i-1', -2: 'Paper i-1', -3:'Scissors i-1',
        -4:'Rock i-2', -5: 'Paper i-2', -6:'Scissors i-2',
        -7:'Rock i-3', -8: 'Paper i-3', -9:'Scissors i-3',
         0:'Rock', 1:'Paper', 2:'Scissors'
    }
    visualize.draw_net(config, winner, True, node_names = node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    print('\nBest genome:\n{!s}'.format(winner))

def simulate_trained_network(sequence_moves, winner, config):
    print('Simulating 100 games with trained network :')
    win = 0
    lose = 0
    draw = 0
    computer_old = [random.randint(0,2), random.randint(0,2), random.randint(0,2)]
    move_number = len(sequence_moves)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    for i in range(1, 101) : 
        computer_old.append(sequence_moves[(i-1) % move_number])
        computer = sequence_moves[i % move_number]
        computer_ohe = np.zeros((3,3))
        computer_ohe[0, computer_old[-1]] = 1
        computer_ohe[1, computer_old[-2]] = 1
        computer_ohe[2, computer_old[-3]] = 1

        # output = winner_net.activate((computer_old[-1], computer_old[-2], computer_old[-3]))
        output = winner_net.activate((
                computer_ohe[0, 0], computer_ohe[0, 1], computer_ohe[0, 2],
                computer_ohe[1, 0], computer_ohe[1, 1], computer_ohe[1, 2], 
                computer_ohe[2, 0], computer_ohe[2, 1], computer_ohe[2, 2] 
        ))
        move = np.argmax(output) 
        results = mat_games[move, computer]
        if results == 0:
            draw += 1
            str_results = 'Draw'
        elif results == 1:
            win += 1
            str_results = 'Win'
        else :
            lose += 1
            str_results = 'Lose'
        #print('{} (IA) VS {} (Computer) ==> {}'.format(idx_moves[move], idx_moves[computer], str_results))   
    
    print('Win : {}, Lose : {}, Draw : {}'.format(win, lose, draw))
    
def simulate_random(sequence_moves):
    print('Simulating 100 random games :')
    win = 0
    lose = 0
    draw = 0
    move_number = len(sequence_moves)

    for i in range(1, 101) : 
        computer = sequence_moves[i % move_number]
        move = random.randint(0, 2)
        results = mat_games[move, computer]
        if results == 0:
            draw += 1
            str_results = 'Draw'
        elif results == 1:
            win += 1
            str_results = 'Win'
        else :
            lose += 1
            str_results = 'Lose'
        #print('{} (IA) VS {} (Computer) ==> {}'.format(idx_moves[move], idx_moves[computer], str_results))   
    
    print('Win : {}, Lose : {}, Draw : {}'.format(win, lose, draw))

def play_trained_network(winner, config):
    win = 0
    lose = 0
    draw = 0
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    player_old = [random.randint(0,2), random.randint(0,2), random.randint(0,2)]

    i = 1
    while True:
        player_move = int(input("Rock (0), Paper (1), Scissors(2) ?"))
        if player_move in range(0, 3):
            player_old.append(player_move)
            output = winner_net.activate((player_old[-1], player_old[-2], player_old[-3]))
            move = np.argmax(output) 
            results = mat_games[player_move, move]
            if results == 0:
                draw += 1
                str_results = 'Draw'
            elif results == 1:
                win += 1
                str_results = 'Win'
            else :
                lose += 1
                str_results = 'Lose'
            print('{} (Player) VS {} (Computer) ==> {}'.format(idx_moves[player_move], idx_moves[move], str_results)) 
        else:
            break
    print('Win : {}, Lose : {}, Draw : {}'.format(win, lose, draw))

def restore_winner(config, winner_filename):
    winner = pickle.load(open(winner_filename, 'rb'))

    return winner

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path)

    winner_filename = 'winner.pkl'

    # Training network 
    training(config, winner_filename)

    # Restore trained network
    winner = restore_winner(config, winner_filename)

    simulate_random(sequence_moves)
    simulate_trained_network(sequence_moves, winner, config)
    # play_trained_network(winner, config)