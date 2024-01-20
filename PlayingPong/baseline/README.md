To run the training on Google Colab, please follow the instructions below:

1. Open the notebook `PlayingPong/baseline/PlayingPong.ipynb` in Google Colab.
2. Run the first cell to install the required dependencies.
3. Run the second cell to start training the agent.

Note: It seems like the training gets interrupted after about 600 episodes. This seems to be related to running out of RAM. You can reload the last saved model and adjust the epsilon to continue training.

For running this on runpod.io

1. Start up a server with PyTorch container and open port 6006
2. Upload the files in the `PlayingPong/baseline` folder to the server
3. Install the following dependencies:
    - `gymnasium[atari, accept-rom-license]`
    - `tensorboard`
    - `opencv-python`
4. Run `tensorboard --logdir=runs` to start TensorBoard
5. Run `python training.py --cuda` to start training the agent
6. Bind the port 6006 to your local machine `ssh root@<server-ip> -p <server-port> -i <path-to-key> -L 6006:localhost:6006`
