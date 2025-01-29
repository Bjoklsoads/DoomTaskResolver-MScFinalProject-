import argparse
import env as grounding_env
from models import A3C_LSTM
import torch
import torch.optim as optim
import os

parser = argparse.ArgumentParser(description='Environment Test')
parser.add_argument('-l', '--max-episode-length', type=int, default=30
                    )
parser.add_argument('-d', '--difficulty', type=str, default="hard"
                    )
parser.add_argument('--living-reward', type=float, default=0
                    )
parser.add_argument('--frame-width', type=int, default=300
                    )
parser.add_argument('--frame-height', type=int, default=168
                    )
parser.add_argument('-v', '--visualize', type=int, default=1
                    )
parser.add_argument('--sleep', type=float, default=0
                    )
parser.add_argument('-t', '--use_train_instructions', type=int, default=1
                    )
parser.add_argument('--scenario-path', type=str, default="maps/room.wad"
                    )
parser.add_argument('--interactive', type=int, default=0
                    )
parser.add_argument('--all-instr-file', type=str,
                    default="data/instructions_all.json"
                    )
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json"
                    )
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json"
                    )


def train_model(env, model, optimizer, num_episodes):
    for episode in range(num_episodes):
        state, reward, is_final, _ = env.reset()
        hx, cx = torch.zeros(1, 256), torch.zeros(1, 256)  # Initialize LSTM hidden states

        total_reward = 0

        while not is_final:
            image, instruction = state
            image = torch.FloatTensor(image).unsqueeze(0)
            instruction = torch.LongTensor([env.word_to_idx[word] for word in instruction.split()]).unsqueeze(0)

            # Get the value, policy, and new LSTM states from the model
            value, policy_logits, (hx, cx) = model(
                (image, instruction, (hx.detach(), cx.detach())))  # Detach hidden states

            # Apply softmax to get a valid probability distribution
            policy_probs = torch.softmax(policy_logits, dim=-1)

            # Sample action from the policy's probability distribution
            action = policy_probs.multinomial(1).detach()

            # Convert action to scalar integer (ensuring compatibility with env.step)
            action_id = action.item()

            # Execute the action in the environment
            state, reward, is_final, _ = env.step(action_id)
            total_reward += reward

            # Reshape value tensor to match the shape of the reward tensor
            value = value.view(-1)  # Reshape value from [1, 1] to [1]

            # Calculate loss
            advantage = reward - value.item()
            policy_loss = -torch.log(policy_probs[0][action_id]) * advantage
            value_loss = torch.nn.functional.smooth_l1_loss(value, torch.tensor([reward]))
            loss = policy_loss + value_loss

            # Backpropagation and optimization
            optimizer.zero_grad()

            # Detach LSTM states to avoid in-place modification errors
            loss.backward(retain_graph=True)

            # Perform the optimization step
            optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

    # Save the trained model after all episodes
    torch.save(model.state_dict(), 'last_trained.pth')
    print("Model saved to 'last_trained.pth'")


def load_trained_model(model_path, args):
    # Instantiate the model architecture
    model = A3C_LSTM(args)

    # Load the saved weights into the model
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    return model


def main(args):
    env = grounding_env.GroundingEnv(args)
    env.game_init()

    # Generate a dummy input to determine input size
    dummy_input = env.reset()[0]

    # Extract image tensor from dummy_input (assuming first element is the image tensor)
    image_tensor = torch.from_numpy(dummy_input[0])
    input_size = image_tensor.shape[0]  # Correctly getting shape of the image tensor

    # Extract instruction string and map it to the corresponding word indices
    instruction_string = dummy_input[1]
    instruction_tensor = torch.LongTensor([env.word_to_idx[word] for word in instruction_string.split()]).unsqueeze(0)

    model_path = 'last_trained.pth'

    # Check if pre-trained model exists
    if os.path.exists(model_path):
        print("Loading pre-trained model from {}".format(model_path))
        trained_model = load_trained_model(model_path, args)
    else:
        print("No pre-trained model found, initializing a new model.")
        args.input_size = input_size  # Set input size for the new model
        trained_model = A3C_LSTM(args)  # Initialize a new model

    # Continue training the model (whether loaded or newly initialized)
    optimizer = optim.Adam(trained_model.parameters(), lr=0.001)

    # Set the number of episodes
    num_episodes = 5

    # Train the model
    train_model(env, trained_model, optimizer, num_episodes)

    print("Training completed.")
    torch.save(trained_model.state_dict(), model_path)  # Save model after training
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
