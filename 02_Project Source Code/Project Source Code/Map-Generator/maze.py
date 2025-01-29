from __future__ import print_function

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('maze')
parser.add_argument(
    '-n',
    '--num',
    type=int,
    default=10
    )
parser.add_argument(
    '-s',
    '--split',
    type=float,
    default=0.8
    )
parser.add_argument(
    '-r', '--rows', type=int, default=9)
parser.add_argument(
    '-c',
    '--columns',
    type=int,
    default=12)

WALL_TYPE = np.int8
WALL = 0
EMPTY = 1

class Maze:
    def __init__(self, rows, columns):
        assert rows >= 1 and columns >= 1

        self.nrows = rows
        self.ncolumns = columns
        self.board = np.zeros((rows, columns), dtype=WALL_TYPE)
        self.board.fill(EMPTY)
        self.player_start = None

    def __str__(self):
        return os.linesep.join(''.join('P' if (i, j) == self.player_start else 'X' if self.is_wall(i, j) else ' '
                                       for j in range(self.ncolumns))
                               for i in range(self.nrows))

    def set_player_start(self, x, y):
        if not self.is_wall(x, y):
            self.player_start = (x, y)
        else:
            raise ValueError("Player start position cannot be inside a wall")

    def __hash__(self):
        return hash(self.board.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def set_borders(self):
        self.board[0, :] = self.board[-1, :] = WALL
        self.board[:, 0] = self.board[:, -1] = WALL

    def is_wall(self, x, y):
        assert self.in_maze(x, y)
        return self.board[x][y] == WALL

    def set_wall(self, x, y):
        assert self.in_maze(x, y)
        self.board[x][y] = WALL

    def remove_wall(self, x, y):
        assert self.in_maze(x, y)
        self.board[x][y] = EMPTY

    def in_maze(self, x, y):
        return 0 <= x < self.nrows and 0 <= y < self.ncolumns

    def write_to_file(self, filename):
        f = open(filename, 'w')
        f.write(str(self))
        f.close()

    @staticmethod
    def create_maze(rows, columns, seed=None, complexity=.5, density=.2):
        rows = (rows // 2) * 2 + 1
        columns = (columns // 2) * 2 + 1

        if seed is not None:
            np.random.seed(seed)

        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (rows + columns)))
        density = int(density * ((rows // 2) * (columns // 2)))

        maze = Maze(rows, columns)
        maze.set_borders()

        # Make aisles
        for i in range(density):
            x = x = np.random.randint(0, rows // 2 + 1) * 2
            y = np.random.randint(0, columns // 2 + 1) * 2
            maze.set_wall(x, y)

            for j in range(complexity):
                neighbours = []

                if maze.in_maze(x - 2, y):
                    neighbours.append((x - 2, y))

                if maze.in_maze(x + 2, y):
                    neighbours.append((x + 2, y))

                if maze.in_maze(x, y - 2):
                    neighbours.append((x, y - 2))

                if maze.in_maze(x, y + 2):
                    neighbours.append((x, y + 2))

                if len(neighbours):
                    next_x, next_y = neighbours[np.random.randint(
                        0,
                        len(neighbours))]

                    if not maze.is_wall(next_x, next_y):
                        maze.set_wall(next_x, next_y)
                        maze.set_wall(next_x + (x - next_x) // 2,
                                      next_y + (y - next_y) // 2)
                        x, y = next_x, next_y

        while True:
            px, py = np.random.randint(0, rows), np.random.randint(0, columns)
            if not maze.is_wall(px, py):
                maze.set_player_start(px, py)
                break

        return maze


if __name__ == '__main__':
    FLAGS = parser.parse_args()

    counter = 0
    mazes = set()

    while len(mazes) < FLAGS.num:
        if counter > 5:
            break

        maze = Maze.create_maze(FLAGS.columns + 1, FLAGS.rows + 1)

        if maze in mazes:
            counter += 1
        else:
            counter = 0
            mazes.add(maze)

    train_indices = set(
        np.random.choice(
            range(len(mazes)),
            size=int(len(mazes) * FLAGS.split),
            replace=False))

    print('{} x {}, {} mazes generated'.format(FLAGS.columns, FLAGS.rows,
                                               len(mazes)))
    print('-- {} train mazes'.format(len(train_indices)))
    print('-- {} test mazes'.format(len(mazes) - len(train_indices)))

    output_dir = "generated map indicator"
    os.makedirs(output_dir, exist_ok=True)

    for idx, maze in enumerate(mazes):
        prefix = 'TRAIN' if idx in train_indices else 'TEST'
        output_filename = "{}_{}_MAP{:02d}.txt".format(FLAGS.maze, prefix, idx + 1)
        output_path = os.path.join(output_dir, output_filename)

        maze.write_to_file(output_path)
        print(f"Saved maze to {output_path}")
