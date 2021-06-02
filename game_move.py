

from maze import Qmaze, show
import numpy as np


def next_move(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    print("next move",envstate)

    n = 0;
    plt, canvas = show(qmaze)
    plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.savefig('./runStep/step' + str(n) + '.png')
    plt.show()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        n+=1
        plt, canvas = show(qmaze)
        plt.imshow(canvas, interpolation='none', cmap='gray')
        plt.savefig('./runStep/step' + str(n) + '.png')
        plt.show()
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