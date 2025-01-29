from __future__ import print_function
from time import sleep

import numpy as np
import collections
import json
import omg
from utils.doom import *
from utils.points import *
from constants import *
import vizdoom

actions = [[True, False, False], [False, True, False], [False, False, True]]

ObjectLocation = collections.namedtuple("ObjectLocation", ["x", "y"])
AgentLocation = collections.namedtuple("AgentLocation", ["x", "y", "theta"])


def get_things_from_wad(wad_path):
    """Reads the THINGS section of the WAD file and returns the types and coordinates of items."""
    wad_path = "maps/room.wad"
    wad = omg.WAD(wad_path)
    things = []
    for map_name in wad.maps:
        wad_map = omg.MapEditor(wad.maps[map_name])  # Use the MapEditor class to parse map data
        for thing in wad_map.things:
            things.append((thing.type, thing.x, thing.y))
    return things


class GroundingEnv:
    def __init__(self, args):
        self.object_coordinates = None
        self.params = args
        # Reading train and test instructions.
        self.train_instructions = self.get_instr(self.params.train_instr_file)
        self.test_instructions = self.get_instr(self.params.test_instr_file)
        if self.params.use_train_instructions:
            self.instructions = self.train_instructions
        else:
            self.instructions = self.test_instructions
        self.word_to_idx = self.get_word_to_idx()
        self.objects, self.object_dict = \
            self.get_all_objects(self.params.all_instr_file)

    def game_init(self):
        game = DoomGame()
        game = set_doom_configuration(game, self.params)
        game.init()
        self.game = game

    def reset(self):
        self.game.new_episode()
        self.time = 0
        self.instruction, instruction_id = self.get_random_instruction()
        print("Retrieved:", self.instruction, instruction_id)

        # Get the generation positions of the agent and objects
        agent_x_coordinate, agent_y_coordinate, \
            agent_orientation, object_x_coordinates, \
            object_y_coordinates, object_types = self.get_agent_and_object_positions()

        # Create a list containing the positions of objects
        self.object_coordinates = [ObjectLocation(x, y) for x, y in
                                   zip(object_x_coordinates,
                                       object_y_coordinates)]

        non_player_indices = [i for i, object_type in enumerate(object_types) if object_type != 1]

        if non_player_indices:
            # Randomly select a non-Type 1 object as the target
            self.correct_location = random.choice(non_player_indices)
            print("Randomly selected target position is:", self.correct_location)
        else:
            raise ValueError("No non-player spawn objects found in the map.")

        # Generate the agent
        spawn_agent(self.game, agent_x_coordinate,
                    agent_y_coordinate, agent_orientation)

        # Generate the objects
        for object_id, pos_x, pos_y in zip(object_types, object_x_coordinates, object_y_coordinates):
            spawn_object(self.game, object_id, pos_x, pos_y)

        screen = self.game.get_state().screen_buffer
        screen_buf = process_screen(screen, self.params.frame_height,
                                    self.params.frame_width)

        state = (screen_buf, self.instruction)
        reward = self.get_reward()
        is_final = False
        extra_args = None

        return state, reward, is_final, extra_args

    def step(self, action_id):
        """Executes an action in the environment to reach a new state.

        Args:
          action_id: An integer corresponding to the action.

        Returns:
           state: A tuple of screen buffer state and instruction.
           reward: Reward at that step.
           is_final: Flag indicating terminal state.
           extra_args: Dictionary of additional arguments/parameters.
        """
        # Repeat the action for X frames.
        if self.params.visualize:
            # Render X frames for better visualization.
            for _ in range(10):
                self.game.make_action(actions[action_id], 5)
                # Slowing down the game for better visualization.
                sleep(self.params.sleep)
        else:
            self.game.make_action(actions[action_id], 5)

        self.time += 1
        reward = self.get_reward()

        # End the episode if episode limit is reached or
        # agent reached an object.
        is_final = True if self.time == self.params.max_episode_length or reward != self.params.living_reward else False
        screen = self.game.get_state().screen_buffer
        screen_buf = process_screen(
            screen, self.params.frame_height, self.params.frame_width)
        state = (screen_buf, self.instruction)

        return state, reward, is_final, None

    def close(self):
        self.game.close()

    def get_agent_current_location(self):
        agent_x = self.game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        agent_y = self.game.get_game_variable(vizdoom.GameVariable.POSITION_Y)
        return agent_x, agent_y

    def get_reward(self):
        self.agent_x, self.agent_y = self.get_agent_current_location()
        print(f"Agent current location is: ({self.agent_x}, {self.agent_y})")

        # Check if the agent has reached the target object
        target_location = self.object_coordinates[self.correct_location]
        print("Object x is", target_location.x, "Object y is", target_location.y)

        dist = get_l2_distance(self.agent_x, self.agent_y, target_location.x, target_location.y)
        print(f"Distance to target: {dist}")

        if dist <= REWARD_THRESHOLD_DISTANCE:
            reward = CORRECT_OBJECT_REWARD
            print(f"Correct item picked up! Reward: {reward}")
            return reward

        # Check if the agent has touched other non-target objects
        for i, object_location in enumerate(self.object_coordinates):
            if i == self.correct_location or (object_location.x == self.agent_x_coordinate and object_location.y == self.agent_y_coordinate):
                continue
            dist = get_l2_distance(self.agent_x, self.agent_y, object_location.x, object_location.y)
            if dist <= REWARD_THRESHOLD_DISTANCE:
                reward = WRONG_OBJECT_REWARD
                print(f"Wrong item picked up! Penalty: {reward}")
                return reward

        # If no object was touched, return the living reward
        reward = self.params.living_reward
        return reward

    def get_agent_and_object_positions(self):
        """Get agent and object positions based on the current map's objects."""
        # Use the omg library to read the THINGS section from the WAD file
        wad_path = "maps/room.wad"  # Replace with your WAD file path
        things = get_things_from_wad(wad_path)

        if not things:
            raise ValueError("No objects found in the current map's THINGS section.")

        object_x_coordinates = []
        object_y_coordinates = []
        object_types = []

        for thing_type, x, y in things:
            object_x_coordinates.append(x)
            object_y_coordinates.append(y)
            object_types.append(thing_type)
            print(f"Read object coordinates and types: (Type: {thing_type}, X: {x}, Y: {y})")

        # Assume the first item is the agent's position
        self.agent_x_coordinate = object_x_coordinates[0]
        self.agent_y_coordinate = object_y_coordinates[0]
        self.agent_orientation = np.random.randint(4)

        print("Agent spawn position is", self.agent_x_coordinate, self.agent_y_coordinate)
        print("Object coordinates are", object_x_coordinates, object_y_coordinates)

        return self.agent_x_coordinate, self.agent_y_coordinate, self.agent_orientation, \
            object_x_coordinates, object_y_coordinates, object_types

    def get_all_objects(self, filename):
        objects = []
        object_dict = {}
        count = 0
        instructions = self.get_instr(filename)
        for instruction_data in instructions:
            object_names = instruction_data['targets']
            for object_name in object_names:
                if object_name not in objects:
                    objects.append(object_name)
                    object_dict[object_name] = count
                    count += 1

        return objects, object_dict

    def get_target_objects(self, instr_id):
        object_names = self.instructions[instr_id]['targets']
        correct_objects = []
        for object_name in object_names:
            correct_objects.append(self.object_dict[object_name])

        return correct_objects

    def get_instr(self, filename):
        with open(filename, 'rb') as f:
            instructions = json.load(f)
        return instructions

    def get_random_instruction(self):
        instruction_id = np.random.randint(len(self.instructions))
        instruction = self.instructions[instruction_id]['instruction']

        return instruction, instruction_id

    def get_word_to_idx(self):
        word_to_idx = {}
        for instruction_data in self.train_instructions:
            instruction = instruction_data['instruction']
            for word in instruction.split(" "):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
        return word_to_idx
