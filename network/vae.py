import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import utils
import config as c

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Define variational autoencoder class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.dim = (32, int(c.height / 4), int(c.width / 4))
        self.n_input = c.n_joints * 2

        # Encoder
        self.e_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e_conv1 = nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.e_conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1))

        # Variational latent variable layers
        self.fc_mu = nn.Linear(int(np.prod(self.dim)), self.n_input)
        self.fc_log_var = nn.Linear(int(np.prod(self.dim)), self.n_input)

        # Decoder
        self.d_fc1 = nn.Linear(self.n_input, int(np.prod(self.dim)))

        self.d_upconv3 = nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2),
                                            padding=(1, 1))
        self.d_batch3 = nn.BatchNorm2d(32)
        self.d_conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1))

        self.d_upconv4 = nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2),
                                            padding=(1, 1))
        self.d_batch4 = nn.BatchNorm2d(32)
        self.d_conv4 = nn.Conv2d(32, 3, (3, 3), stride=(1, 1), padding=(1, 1))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def encode(self, x):
        """
        Run the encoder
        :param x: input image
        :return: mu and log_var vector
        """
        out = self.relu(self.e_conv1(x))
        out = self.e_pool(out)

        out = self.relu(self.e_conv2(out))
        out = self.e_pool(out)

        out = out.reshape(out.size(0), -1)

        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)

        return mu, log_var

    @staticmethod
    def reparametrize(mu, log_var):
        """
        Randomly sample z based on mu and log_var vectors
        :param mu: latent mean joint angle vector
        :param log_var: latent log variance joint angle vector
        :return: z
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        """
        Run the decoder
        :param z: latent variable vector z
        :return: output image
        """
        out = self.relu(self.d_fc1(z))

        out = out.reshape(-1, *self.dim)

        out = self.relu(self.d_batch3(self.d_upconv3(out)))
        out = self.relu(self.d_conv3(out))

        out = self.relu(self.d_batch4(self.d_upconv4(out)))
        out = self.sigmoid(self.d_conv4(out))

        return out

    def forward(self, x):
        """
        Perform forward pass through the network
        :param x: input image
        :return: output image, mu and log_var vectors
        """
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        output = self.decode(z)

        return output, mu, log_var

    def predict_visual(self, x):
        """
        Get visual prediction
        :param x: input joint
        :return: output image
        """
        input_ = torch.tensor(x, device=device, dtype=torch.float,
                              requires_grad=True)
        output = self.decode(input_)

        return input_, output

    def predict_joint(self, x):
        """
        Get joint prediction
        :param x: input image
        :return: output joint
        """
        input_ = torch.tensor(x, device=device, dtype=torch.float).unsqueeze(0)
        output, _ = self.encode(input_)

        return output

    def get_grad(self, input_, output, error):
        """
        Get gradient with respect to prediction error
        :param input_: input tensor from the predict function
        :param output: output tensor from the predict function
        :param error: prediction error
        :return: numpy array containing the gradient
        """
        # Set gradient to zero
        input_.grad = torch.zeros(input_.size(), device=device,
                                  dtype=torch.float, requires_grad=False)

        output.backward(torch.tensor(error, dtype=torch.float,
                                     device=device))

        return input_.grad.detach().cpu().numpy()

    @staticmethod
    def train_net(net, x, y):
        """
        Train the neural network
        :param net: network object
        :param x: input samples
        :param y: output samples
        """
        torch.cuda.empty_cache()

        # Split dataset
        train_gen, test_gen = utils.split_dataset(x, y, 0.9)

        # Define optimizer
        optimizer = optim.Adam(net.parameters(), lr=c.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=c.step_size,
                                              gamma=c.gamma)

        train_loss, test_loss, mse_loss = [], [], []

        # Start training
        for epoch in range(c.n_epochs):
            cur_train_loss = 0
            for x_train, y_train in train_gen:
                loss, _ = VAE.run_batch(x_train, y_train, True, net, optimizer)
                cur_train_loss += loss
            train_loss.append(cur_train_loss / len(train_gen))

            scheduler.step()

            # Evaluate network
            if (epoch + 1) % 5 == 0:
                cur_test_loss, cur_mse_loss = 0, 0
                for x_test, y_test in test_gen:
                    loss, mse = VAE.run_batch(x_test, y_test, False, net,
                                              optimizer)
                    cur_test_loss += loss
                    cur_mse_loss += mse
                test_loss.append(cur_test_loss / len(test_gen))
                mse_loss.append(cur_mse_loss / len(test_gen))

                sys.stdout.write('\rEpoch {:3d}  ===> \t Train loss: {:10.5f}'
                                 ' \t Test loss: {:10.5f} \t\t MSE: {:10.5f}'.
                                 format(epoch + 1, train_loss[-1],
                                        test_loss[-1], mse_loss[-1]))
                sys.stdout.flush()

    @staticmethod
    def run_batch(x_target, y_target, train, net, optimizer):
        """
        Execute a training batch
        :param x_target: input samples
        :param y_target: output samples
        :param train: variable for network training
        :param net: network object
        :param optimizer: optimizer
        """
        # Send to GPU
        x_target = x_target.to(device, torch.float32)
        y_target = y_target.to(device, torch.float32)

        # Clear gradient
        optimizer.zero_grad()

        # Forward pass
        y_predict, mu, log_var = net(y_target)

        # Compute loss
        loss, mse = VAE.loss_function(y_predict, y_target, x_target,
                                      mu, log_var)

        if train:
            # Perform optimization step
            loss.backward()
            optimizer.step()

        return loss.item(), mse.item()

    @staticmethod
    def loss_function(y_predict, y_target, x_target, mu, log_var):
        """
        Loss function of the VAE, based on the MSE of the images and
        regularization term based on the Kullback-Leibler divergence
        :param y_predict: predicted image
        :param y_target: target image
        :param x_target: ground truth joint angles
        :param mu: latent mean joint angle vector
        :param log_var: latent log variance joint angle vector
        :return: loss
        """
        criterion_mse = nn.MSELoss()

        mse = criterion_mse(y_predict, y_target)

        var_target = torch.full(log_var.shape, c.variance).to(
            device, torch.float32)
        kld = utils.kl_divergence(x_target, var_target, mu, log_var)

        return mse + kld, mse

    def save(self):
        """
        Save network to file
        """
        torch.save(self.state_dict(), 'network/vae.net')

    def load(self):
        """
        Load network from file
        """
        self.load_state_dict(torch.load('network/vae.net',
                                        map_location=device))
        self.eval()
