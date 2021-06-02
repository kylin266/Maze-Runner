from trainer import *

maze = np.array([[1., 0., 1., 1., 1., 1., 1.], [1., 1., 1., 0., 0., 1., 0.],
                 [0., 0., 0., 1., 1., 1., 0.], [1., 1., 1., 1., 0., 0., 1.],
                 [1., 0., 0., 0., 1., 1., 1.], [1., 0., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 0., 1., 1., 1.]])

model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8 * maze.size, data_size=32)
