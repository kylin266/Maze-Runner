from maze import Qmaze
import numpy as np

maze = [[1., 0., 1., 1., 1., 1., 1., 1.], [1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 0., 1., 0., 1.], [1., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1., 1., 1.], [1., 1., 1., 0., 1., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 1., 1.], [1., 1., 1., 1., 0., 1., 1., 1.]]


def next_move(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False
def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not next_move(model, qmaze, cell):
            return False
    return True