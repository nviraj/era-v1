import torch


def get_device():
    """
    Function to get the device to be used for training and testing.
    """

    # Check if cuda is available
    cuda = torch.cuda.is_available()

    # Based on check enable cuda if present, if not available
    if cuda:
        final_choice = "cuda"
    else:
        final_choice = "cpu"

    return final_choice, torch.device(final_choice)


def get_correct_prediction_count(pPrediction, pLabels):
    """
    Function to get the count of correct predictions.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
