
from thegame.game_move import *
from maze import *
from trainer import build_model

# load weights into new model


maze = np.array([[1., 0., 1., 1., 1., 1., 1.], [1., 1., 1., 0., 0., 1., 0.],
                 [0., 0., 0., 1., 1., 1., 0.], [1., 1., 1., 1., 0., 0., 1.],
                 [1., 0., 0., 0., 1., 1., 1.], [1., 0., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 0., 1., 1., 1.]])

model = build_model(maze)
model.load_weights("../model/model.h5")
qmaze = Qmaze(maze);



def play(f):
    game_over = False
    n_episodes = 0
    while not game_over:
        game_over = next_move(f, qmaze,[0,0])


play(model)
