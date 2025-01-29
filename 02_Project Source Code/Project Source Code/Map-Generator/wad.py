import random
import re
from omg import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('prefix')
parser.add_argument('wad')
parser.add_argument(
    '-b',
    '--behavior',
    default=False,
    )
parser.add_argument(
    '-s',
    '--script',
    default=False,
    )

BLOCK_SIZE = 96

common_items = [
    (82, "Shotgun"), (2003, "Rocket Launcher"), (2004, "Plasma Gun"),
    (2005, "Chainsaw"), (2006, "BFG9000"), (2019, "Blue Armor"),(2007, "Ammo Clip"),
    (2010, "Rocket"), (13, "Red Keycard"), (6, "Yellow Keycard"), (5, "Blue Keycard"),
    (40, "Blue Skull Keycard"), (39, "Yellow Skull Keycard"),  (38, "RED Skull Keycard"),
]


def build_wall(maze):
    things = []
    linedefs = []
    vertexes = []
    v_indexes = {}

    max_w = len(maze[0]) - 1
    max_h = len(maze) - 1

    player_start_set = False

    def __is_edge(w, h):
        return w in (0, max_w) or h in (0, max_h)

    def __add_start(w, h):
        if not player_start_set:  # Only if player start position is not set.
            x, y = w * BLOCK_SIZE, h * BLOCK_SIZE
            x += int(BLOCK_SIZE / 2)
            y += int(BLOCK_SIZE / 2)
            print(f"Player start coordinates: ({x}, {y})")
            things.append(ZThing(*[len(things) + 1000, x, y, 0, 0, 1, 7]))  # set player 1's spawn point
            return True
        return False

    def __add_vertex(w, h):
        if (w, h) in v_indexes:
            return

        x, y = w * BLOCK_SIZE, h * BLOCK_SIZE
        x += int(BLOCK_SIZE / 2)
        y += int(BLOCK_SIZE / 2)
        v_indexes[w, h] = len(vertexes)
        vertexes.append(Vertex(x, y))

    def __add_line(start, end, edge=False):
        assert start in v_indexes
        assert end in v_indexes

        mask = 1
        left = right = 0
        if __is_edge(*start) and __is_edge(*end):
            if not edge:
                return
            else:
                # Changed the right side (one towards outside the map).
                # to be -1 (65535 for Doom).
                right = 65535
                mask = 15

        # Flipped end and start vertices to make lines "point" at right direction.
        line_properties = [v_indexes[end], v_indexes[start], mask
                           ] + [0] * 6 + [left, right]
        line = ZLinedef(*line_properties)
        linedefs.append(line)

    for h, row in enumerate(maze):
        for w, block in enumerate(row.strip()):
            if block == 'X':
                __add_vertex(w, h)
            elif block == 'P' and not player_start_set:
                __add_start(w, h)
                player_start_set = True  # Update flags to avoid multiple settings.
            else:
                pass

    corners = [(0, 0), (max_w, 0), (max_w, max_h), (0, max_h)]
    for v in corners:
        __add_vertex(*v)

    for i in range(len(corners)):
        if i != len(corners) - 1:
            __add_line(corners[i], corners[i + 1], True)
        else:
            __add_line(corners[i], corners[0], True)

    # Now connect the walls
    for h, row in enumerate(maze):
        for w, _ in enumerate(row):
            if (w, h) not in v_indexes:
                continue

            if (w + 1, h) in v_indexes:
                __add_line((w, h), (w + 1, h))

            if (w, h + 1) in v_indexes:
                __add_line((w, h), (w, h + 1))

    # print("Things:", things)  # print things
    # print("Vertexes:", vertexes)  # print vertexes
    # print("Linedefs:", linedefs)  # print linedefs

    return things, vertexes, linedefs


def is_wall(maze, x, y):
    maze_x = x // BLOCK_SIZE
    maze_y = y // BLOCK_SIZE
    return maze[maze_y][maze_x] == 'X'


def main(flags):
    new_wad = WAD()

    # Access to all eligible schematic documents.
    file_names = glob.glob(os.path.join('generated map indicator', '{}_*.txt'.format(flags.prefix)))
    if file_names:
        # Randomly choose only one map indicator file.
        file_name = random.choice(file_names)
        with open(file_name) as maze_source:
            maze = [line.strip() for line in maze_source.readlines()]
            maze = [line for line in maze if line]
        print(f"Selected maze file: {file_name}")

        new_map = MapEditor()
        new_map.Linedef = ZLinedef
        new_map.Thing = ZThing
        new_map.behavior = Lump(from_file=flags.behavior or None)
        new_map.scripts = Lump(from_file=flags.script or None)
        things, vertexes, linedefs = build_wall(maze)

        # Add Doom items to THINGS and unique ids for items
        max_x = len(maze[0]) * BLOCK_SIZE
        max_y = len(maze) * BLOCK_SIZE
        unique_id = 1
        for item_id, item_name in common_items:
            x = random.randint(0, len(maze[0]) - 1) * BLOCK_SIZE + BLOCK_SIZE
            y = random.randint(0, len(maze) - 1) * BLOCK_SIZE + BLOCK_SIZE
            if x < max_x and y < max_y:
                print(f"Adding {item_name} (ID: {item_id}) at ({x}, {y})")
                things.append(ZThing(unique_id, x, y, 0, 0, item_id, 7 + 256, 0, 0, 0, 0, 0, 0))
                unique_id += 1
            else:
                print(f"Re-adding {item_name} (ID: {item_id}) at ({x}, {y}) -  It was out of bounds")

        new_map.things = things
        new_map.vertexes = vertexes
        new_map.linedefs = linedefs
        new_map.sectors = [Sector(0, 128, 'CEIL5_2', 'CEIL5_2', 240, 0, 0)]
        # Sidedefs define the texture of the wall
        new_map.sidedefs = [
            Sidedef(0, 0, '-', '-', 'STONE2', 0),
            Sidedef(0, 0, '-', '-', '-', 0)
        ]
        new_wad.maps['MAP01'] = new_map.to_lumps()  # Only generate 1 map which is MAP01,
        # cause MAP00 doesn't seem to load properly

        # Save to folder map
        output_dir = "generated map"
        os.makedirs(output_dir, exist_ok=True)  # If the folder does not exist, create the map folder
        output_path = os.path.join(output_dir, flags.wad)
        new_wad.to_file(output_path)  # Save the wad into the folder with wad format



if __name__ == "__main__":
    main(parser.parse_args())
