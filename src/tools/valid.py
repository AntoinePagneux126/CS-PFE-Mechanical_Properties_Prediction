"""This module define the function to test the model on one epoch."""
# pylint: disable=import-error, no-name-in-module
import os

import numpy
import torch
import tqdm


from sklearn.metrics import r2_score


class ModelCheckpoint:  # pylint: disable=too-few-public-methods

    """Define the model checkpoint class
    """

    def __init__(self, dir_path, model, epochs, checkpoint_step):
        self.min_loss = None
        self.dir_path = dir_path
        self.best_model_filepath = os.path.join(self.dir_path, "best_model.pth")
        self.model = model
        self.epochs = epochs
        self.checkpoint_step = checkpoint_step

    def update(self, loss, epoch):
        """Update the model if the we get a smaller lost

        Args:
            loss (float): Loss over one epoch
        """

        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.best_model_filepath)
            self.min_loss = loss

        if epoch in numpy.arange(
            self.checkpoint_step - 1, self.epochs, self.checkpoint_step
        ):

            print(f"Saving model at Epoch {epoch}")

            filename = "epoch_" + str(epoch) + "_model.pth"

            filepath = os.path.join(self.dir_path, filename)
            torch.save(self.model.state_dict(), filepath)


def test_one_epoch(model, loader, f_loss, device, return_predictions=False):
    """Test the model for one epoch

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        device (torch.device): cuda
        return_predictions (bool): Should you return predicted val and target

    Returns:
        tot_loss/N (float) : accumulated loss over one epoch
        r2_score (float) : r2 score over one epoch
    """

    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        n_samples = 0
        tot_loss = 0.0
        predicted_targets_all, targets_all = None, None

        for inputs, targets in tqdm.tqdm(loader):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = torch.flatten(model(inputs))

            # We accumulate the exact number of processed samples
            n_samples += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores

            # Concat the result in order to compute f1-score
            if predicted_targets_all is None:
                predicted_targets_all = outputs
                targets_all = targets
            else:
                predicted_targets_all = torch.cat((predicted_targets_all, outputs))
                targets_all = torch.cat((targets_all, targets))

        r2_score_ = r2_score(
            y_true=targets_all.cpu().int().numpy(),
            y_pred=predicted_targets_all.cpu().int().numpy(),
        )

        if not return_predictions:
            return tot_loss / n_samples, r2_score_

        return (
            tot_loss / n_samples,
            r2_score_,
            targets_all.cpu().int().numpy(),
            predicted_targets_all.cpu().int().numpy(),
        )
